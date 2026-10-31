use super::{emit::Emitter, invalid, types::Scalar, Result};
use ruda_core::ir::{Branch, CoopMma, ElemType, FloatKind, Metadata, NonSemantic, Operation, OperationReflect, Operator, Scope, Variable, VariableKind};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct Arrays {
    candidates: HashMap<u32, Variable>,
    escaped: HashSet<u32>,
    unpaired: HashSet<u32>,
    unknown: bool,
}

impl Arrays {
    fn record(&mut self, variable: Variable, allowed: bool) {
        if let VariableKind::LocalArray { id, length, unroll_factor } = variable.kind {
            let count = length.checked_mul(variable.ty.vector_size());
            if !allowed || unroll_factor != 1 || !count.is_some_and(|n| n > 0 && n <= 256) {
                self.escaped.insert(id);
            }
            if let Some(previous) = self.candidates.insert(id, variable) {
                if previous != variable { self.escaped.insert(id); }
            }
        }
    }

    fn indexed(&mut self, array: Variable, index: Variable, width: usize, unroll: usize) {
        let width = if width == 0 { array.ty.vector_size() } else { width };
        if width % 2 != 0 && let VariableKind::LocalArray { id, .. } = array.kind { self.unpaired.insert(id); }
        let in_bounds = match (array.kind, index.as_const()) {
            (VariableKind::LocalArray { length, .. }, Some(index)) => {
                index.as_u64().checked_add(1).and_then(|n| n.checked_mul(width as u64))
                    .zip(length.checked_mul(array.ty.vector_size()))
                    .is_some_and(|(end, capacity)| end <= capacity as u64)
            }
            _ => false,
        };
        self.record(array, in_bounds && unroll == 1 && index.ty.is_unsigned_int() && index.ty.vector_size() == 1);
    }

    fn scope(&mut self, scope: &Scope) {
        for parameter in &scope.parameters { self.record(*parameter, false); }
        for instruction in &scope.instructions {
            let Ok(instruction) = instruction.clone().lower_typed_memory() else { self.unknown = true; continue; };
            match &instruction.operation {
                Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op)) => {
                    self.indexed(op.list, op.index, op.vector_size, op.unroll_factor);
                    if let Some(out) = instruction.out { self.record(out, false); }
                }
                Operation::Operator(Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op)) => {
                    if let Some(out) = instruction.out { self.indexed(out, op.index, op.vector_size, op.unroll_factor); }
                    self.record(op.value, false);
                }
                Operation::CoopMma(CoopMma::ExecuteManual { registers_a, registers_b, registers_c, .. }) => {
                    for variable in [*registers_a, *registers_b, *registers_c].into_iter().chain(instruction.out) {
                        self.record(variable, true);
                    }
                }
                Operation::CoopMma(CoopMma::ExecuteScaled { registers_a, registers_b, registers_c, scales_a, scales_b, .. }) => {
                    for variable in [*registers_a, *registers_b, *registers_c, *scales_a, *scales_b].into_iter().chain(instruction.out) {
                        self.record(variable, true);
                    }
                }
                Operation::CoopMma(CoopMma::Blackwell { kind, args }) => {
                    for arg in args { self.record(*arg, false); }
                    if let Some(out) = instruction.out {
                        self.record(out, matches!(kind, ruda_core::ir::Tcgen05Op::Load { .. }));
                    }
                }
                Operation::CoopMma(CoopMma::LoadMatrix { buffer, offset, .. }) => {
                    self.record(*buffer, false); self.record(*offset, false);
                    if let Some(out) = instruction.out { self.record(out, true); }
                }
                Operation::CoopMma(CoopMma::StoreMatrix { registers, offset, .. }) => {
                    self.record(*registers, true); self.record(*offset, false);
                    if let Some(out) = instruction.out { self.record(out, false); }
                }
                Operation::CoopMma(CoopMma::RowIndex { lane_id, i, .. } | CoopMma::ColIndex { lane_id, i, .. }) => {
                    for v in [*lane_id, *i].into_iter().chain(instruction.out) { self.record(v, false); }
                }
                Operation::Metadata(Metadata::Length { var } | Metadata::BufferLength { var }) => {
                    self.record(*var, true);
                }
                Operation::Branch(branch) => match branch {
                    Branch::If(b) => { self.record(b.cond, false); self.scope(&b.scope); }
                    Branch::IfElse(b) => { self.record(b.cond, false); self.scope(&b.scope_if); self.scope(&b.scope_else); }
                    Branch::Switch(b) => {
                        self.record(b.value, false); self.scope(&b.scope_default);
                        for (value, scope) in &b.cases { self.record(*value, false); self.scope(scope); }
                    }
                    Branch::RangeLoop(b) => {
                        for v in [b.i, b.start, b.end].into_iter().chain(b.step) { self.record(v, false); }
                        self.scope(&b.scope);
                    }
                    Branch::Loop(b) => self.scope(&b.scope),
                    Branch::ReturnValue(v) => self.record(*v, false),
                    Branch::Break | Branch::Return | Branch::Unreachable => {}
                },
                Operation::NonSemantic(NonSemantic::Print { args, .. }) => {
                    for v in args { self.record(*v, false); }
                }
                Operation::NonSemantic(_) | Operation::Synchronization(_) => {}
                Operation::Marker(ruda_core::ir::Marker::Free(v)) => self.record(*v, false),
                Operation::Barrier(ruda_core::ir::BarrierOps::CopyAsync { source, source_length, offset_source, offset_out, .. }) => {
                    for v in [*source, *source_length, *offset_source, *offset_out].into_iter().chain(instruction.out) {
                        self.record(v, false);
                    }
                }
                operation => {
                    if let Some(out) = instruction.out { self.record(out, false); }
                    if let Some(args) = operation.args() {
                        for v in args { self.record(v, false); }
                    } else { self.unknown = true; }
                }
            }
        }
    }
}

pub(super) fn eligible(scope: &Scope) -> (HashMap<u32, Variable>, HashSet<u32>) {
    let mut arrays = Arrays::default();
    arrays.scope(scope);
    if arrays.unknown { return (HashMap::new(), HashSet::new()); }
    arrays.candidates.retain(|id, _| !arrays.escaped.contains(id));
    let packed = arrays.candidates.iter().filter_map(|(&id, variable)| {
        (variable.ty.vector_size() % 2 == 0 && !arrays.unpaired.contains(&id)
            && matches!(variable.elem_type(), ElemType::Float(FloatKind::F16 | FloatKind::BF16))).then_some(id)
    }).collect();
    (arrays.candidates, packed)
}

impl Emitter {
    pub(super) fn packed_array_registers(&mut self, array: Variable) -> Result<Option<Vec<String>>> {
        let VariableKind::LocalArray { id, length, .. } = array.kind else { return Ok(None); };
        if !self.packed_register_arrays.contains(&id) { return Ok(None); }
        if self.register_arrays.get(&id) != Some(&array) { return Err(invalid("packed register array declaration mismatch")); }
        if let Some(registers) = self.packed_array_registers.get(&id) { return Ok(Some(registers.clone())); }
        self.half_target(Scalar::of(array.ty.with_vector_size(1))?)?;
        let registers = (0..length * array.ty.vector_size() / 2).map(|_| self.reg_b32()).collect::<Vec<_>>();
        self.packed_array_registers.insert(id, registers.clone());
        Ok(Some(registers))
    }

    pub(super) fn array_registers(&mut self, array: Variable) -> Result<Option<Vec<String>>> {
        let VariableKind::LocalArray { id, length, .. } = array.kind else { return Ok(None); };
        let Some(declared) = self.register_arrays.get(&id) else { return Ok(None); };
        if *declared != array { return Err(invalid("register array declaration mismatch")); }
        if self.packed_register_arrays.contains(&id) { return Err(invalid("packed register array accessed as scalar registers")); }
        if let Some(registers) = self.array_registers.get(&id) { return Ok(Some(registers.clone())); }
        let ty = Scalar::of(array.ty.with_vector_size(1))?;
        self.half_target(ty)?;
        let registers = (0..length * array.ty.vector_size()).map(|_| self.reg(ty)).collect::<Vec<_>>();
        self.array_registers.insert(id, registers.clone());
        Ok(Some(registers))
    }

    pub(super) fn register_array_memory(&mut self, array: Variable, index: Variable, value: Variable, width: usize, store: bool) -> Result<bool> {
        let packed = self.packed_array_registers(array)?;
        let is_packed = packed.is_some();
        let registers = if let Some(registers) = packed { registers }
            else if let Some(registers) = self.array_registers(array)? { registers }
            else { return Ok(false); };
        let index = index.as_const().ok_or_else(|| invalid("dynamic register array index"))?.as_u64();
        let start = usize::try_from(index).ok().and_then(|i| i.checked_mul(width))
            .ok_or_else(|| invalid("register array index overflow"))?;
        let capacity = registers.len() * if is_packed { 2 } else { 1 };
        let end = start.checked_add(width).filter(|end| *end <= capacity)
            .ok_or_else(|| invalid("register array index out of bounds"))?;
        if array.storage_type() != value.storage_type() || value.ty.vector_size() != width {
            return Err(invalid("register array element type mismatch"));
        }
        let ty = Scalar::of(value.ty.with_vector_size(1))?;
        let values = if store { self.vector_values(value)? } else { self.vector_destination(value)? };
        if is_packed {
            if start % 2 != 0 || width % 2 != 0 { return Err(invalid("unpaired access to packed register array")); }
            for (register, pair) in registers[start/2..end/2].iter().zip(values.chunks_exact(2)) {
                if store {
                    let low = self.reg_b16();let high = self.reg_b16();
                    self.line(format!("mov.b16 {low}, {};", pair[0]));
                    self.line(format!("mov.b16 {high}, {};", pair[1]));
                    self.line(format!("mov.b32 {register}, {{{low}, {high}}};"));
                } else {
                    self.line(format!("mov.b32 {{{}, {}}}, {register};", pair[0], pair[1]));
                }
            }
            return Ok(true);
        }
        for (register, value) in registers[start..end].iter().zip(values) {
            let (destination, source) = if store { (register.as_str(), value.as_str()) } else { (value.as_str(), register.as_str()) };
            self.line(format!("mov.{} {destination}, {source};", ty.storage()));
        }
        Ok(true)
    }
}
