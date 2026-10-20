use std::collections::HashSet;

use ruda_core::ir::*;

pub(super) fn scalar_addressed_arrays(scope: &Scope) -> HashSet<Id> {
    fn collect(scope: &Scope, arrays: &mut HashSet<Id>) {
        for inst in &scope.instructions {
            match &inst.operation {
                Operation::Operator(Operator::WorkgroupUniformLoad(op)) => {
                    if let VariableKind::SharedArray { id, .. } = op.list.kind {
                        let stored = op.list.vector_size();
                        let width = if op.vector_size == 0 { stored } else { op.vector_size };
                        if stored > 1 && (width != stored || op.list.ty.is_atomic()) {
                            arrays.insert(id);
                        }
                    }
                }
                Operation::Branch(branch) => match branch {
                    Branch::If(b) => collect(&b.scope, arrays),
                    Branch::IfElse(b) => {
                        collect(&b.scope_if, arrays);
                        collect(&b.scope_else, arrays);
                    }
                    Branch::Switch(b) => {
                        collect(&b.scope_default, arrays);
                        for (_, scope) in &b.cases {
                            collect(scope, arrays);
                        }
                    }
                    Branch::RangeLoop(b) => collect(&b.scope, arrays),
                    Branch::Loop(b) => collect(&b.scope, arrays),
                    _ => {}
                },
                Operation::CoopMma(CoopMma::Elementwise { function, .. }) => {
                    collect(&function.body, arrays);
                }
                _ => {}
            }
        }
    }
    let mut arrays = HashSet::new();
    collect(scope, &mut arrays);
    arrays
}

pub(super) fn lower_scalar_shared_arrays(
    mut processing: ScopeProcessing,
    allocator: &Allocator,
    arrays: &HashSet<Id>,
) -> ScopeProcessing {
    if arrays.is_empty() {
        return processing;
    }
    let mut lowering = Lowering { allocator, arrays, instructions: Vec::new() };
    for var in &mut processing.variables {
        *var = lowering.storage(*var);
    }
    for inst in processing.instructions {
        let start = lowering.instructions.len();
        lowering.instruction(&inst);
        for emitted in &mut lowering.instructions[start..] {
            emitted.source_loc = inst.source_loc.clone();
            emitted.modes = inst.modes;
        }
    }
    processing.instructions = lowering.instructions;
    processing
}

struct Lowering<'a> {
    allocator: &'a Allocator,
    arrays: &'a HashSet<Id>,
    instructions: Vec<Instruction>,
}

impl Lowering<'_> {
    fn selected(&self, var: Variable) -> bool {
        matches!(var.kind, VariableKind::SharedArray { id, .. } if self.arrays.contains(&id))
    }

    fn storage(&self, mut var: Variable) -> Variable {
        if self.selected(var) {
            let width = var.vector_size();
            if let VariableKind::SharedArray { length, .. } = &mut var.kind {
                *length = length.checked_mul(width).expect("shared array scalar length overflow");
            }
            var.ty = var.ty.with_vector_size(1);
        }
        var
    }

    fn emit(&mut self, op: impl Into<Operation>, out: Variable) {
        self.instructions.push(Instruction::new(op, out));
    }

    fn binary(&mut self, lhs: Variable, rhs: Variable, multiply: bool) -> Variable {
        let out = *self.allocator.create_local(lhs.ty);
        let op = BinaryOperator { lhs, rhs };
        self.emit(if multiply { Arithmetic::Mul(op) } else { Arithmetic::Add(op) }, out);
        out
    }

    fn add(&mut self, index: Variable, offset: usize) -> Variable {
        if offset == 0 {
            index
        } else {
            self.binary(index, index.ty.constant(offset.into()), false)
        }
    }

    fn start(&mut self, index: Variable, width: usize) -> Variable {
        if width == 1 {
            index
        } else {
            self.binary(index, index.ty.constant(width.into()), true)
        }
    }

    fn read(&mut self, op: &IndexOperator, out: Variable, uniform: bool) {
        let width = if op.vector_size == 0 { op.list.vector_size() } else { op.vector_size };
        assert_eq!(width, out.vector_size());
        assert!(!out.ty.is_atomic() || width == 1, "WGSL atomic references must be scalar");
        let list = self.storage(op.list);
        let start = self.start(op.index, width);
        let mut inputs = Vec::with_capacity(width);
        for lane in 0..width {
            let value = if width == 1 { out } else {
                *self.allocator.create_local(out.ty.with_vector_size(1))
            };
            let index = self.add(start, lane);
            let index_op = IndexOperator {
                list, index, vector_size: 0, unroll_factor: op.unroll_factor,
            };
            self.emit(if uniform {
                Operator::WorkgroupUniformLoad(index_op)
            } else {
                Operator::UncheckedIndex(index_op)
            }, value);
            inputs.push(value);
        }
        if width > 1 {
            self.emit(Operator::InitVector(VectorInitOperator { inputs }), out);
        }
    }

    fn write(&mut self, op: &IndexAssignOperator, out: Variable) {
        let width = if op.vector_size == 0 { out.vector_size() } else { op.vector_size };
        assert!(op.value.vector_size() == 1 || op.value.vector_size() == width);
        let list = self.storage(out);
        let start = self.start(op.index, width);
        for lane in 0..width {
            let value = if op.value.vector_size() == 1 {
                op.value
            } else {
                let value = *self.allocator.create_local(op.value.ty.with_vector_size(1));
                self.emit(Operator::UncheckedIndex(IndexOperator {
                    list: op.value,
                    index: (lane as u32).into(),
                    vector_size: 0,
                    unroll_factor: 1,
                }), value);
                value
            };
            let index = self.add(start, lane);
            self.emit(Operator::UncheckedIndexAssign(IndexAssignOperator {
                index, value, vector_size: 0, unroll_factor: op.unroll_factor,
            }), list);
        }
    }

    fn copy(&mut self, input: Variable, in_index: Variable, out: Variable, out_index: Variable) {
        assert_eq!(input.ty, out.ty, "memory copies preserve element type and vector width");
        assert!(!input.ty.is_atomic(), "WGSL atomic storage must be accessed with atomic operations");
        let value = *self.allocator.create_local(input.ty);
        let read = IndexOperator {
            list: input, index: in_index, vector_size: 0, unroll_factor: 1,
        };
        if self.selected(input) {
            self.read(&read, value, false);
        } else {
            self.emit(Operator::UncheckedIndex(read), value);
        }
        let write = IndexAssignOperator {
            index: out_index, value, vector_size: 0, unroll_factor: 1,
        };
        if self.selected(out) {
            self.write(&write, out);
        } else {
            self.emit(Operator::UncheckedIndexAssign(write), out);
        }
    }

    fn instruction(&mut self, original: &Instruction) {
        let inst = original.clone().lower_typed_memory().expect("valid typed memory instruction");
        match &inst.operation {
            Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op))
                if self.selected(op.list) => self.read(op, inst.out(), false),
            Operation::Operator(Operator::WorkgroupUniformLoad(op))
                if self.selected(op.list) => self.read(op, inst.out(), true),
            Operation::Operator(Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op))
                if self.selected(inst.out()) => self.write(op, inst.out()),
            Operation::Operator(Operator::CopyMemory(op))
                if self.selected(op.input) || self.selected(inst.out()) => {
                    self.copy(op.input, op.in_index, inst.out(), op.out_index);
                }
            Operation::Operator(Operator::CopyMemoryBulk(op))
                if self.selected(op.input) || self.selected(inst.out()) => {
                    if op.len == 0 {
                        return;
                    }
                    let input = self.binary(op.in_index, op.offset_input, false);
                    let output = self.binary(op.out_index, op.offset_out, false);
                    for line in 0..op.len {
                        let in_index = self.add(input, line);
                        let out_index = self.add(output, line);
                        self.copy(op.input, in_index, inst.out(), out_index);
                    }
                }
            Operation::Metadata(Metadata::Length { var } | Metadata::BufferLength { var })
                if self.selected(*var) => {
                    let VariableKind::SharedArray { length, .. } = var.kind else { unreachable!() };
                    self.emit(Operation::Copy(inst.out().ty.constant(length.into())), inst.out());
                }
            Operation::Marker(Marker::Free(var)) if self.selected(*var) => {
                self.instructions.push(Instruction::no_out(Marker::Free(self.storage(*var))));
            }
            _ => self.instructions.push(original.clone()),
        }
    }
}
