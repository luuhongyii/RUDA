use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{AtomicOp, Instruction, Operation, OperationReflect, Operator, StorageType, Type, Variable, VariableKind};

mod pointer;
mod reduction;
mod integer;
mod vector;

fn is_atomic(variable: Variable) -> bool {
    matches!(variable.ty, Type::Scalar(StorageType::Atomic(_)) | Type::Vector(StorageType::Atomic(_), _))
}

fn element(variable: Variable) -> Result<Scalar> {
    let (Type::Scalar(StorageType::Atomic(elem)) | Type::Vector(StorageType::Atomic(elem), _)) = variable.ty else {
        return Err(unsupported("atomic pointer must have a scalar atomic element"));
    };
    let ty = Scalar::of(Type::scalar(elem))?;
    if matches!(ty, Scalar::I8 | Scalar::U8) { return Err(unsupported("8-bit integer atomics")); }
    if ty == Scalar::Pred { return Err(invalid("predicate atomic element")); }
    Ok(ty)
}

impl Emitter {
    pub(super) fn workgroup_uniform_load(
        &mut self, op: &ruda_core::ir::IndexOperator, out: Variable,
    ) -> Result<()> {
        if !matches!(op.list.kind, VariableKind::SharedArray { .. } | VariableKind::Shared { .. }) {
            return Err(invalid("workgroup uniform load requires shared memory"));
        }
        let width = if op.vector_size == 0 { op.list.ty.vector_size() } else { op.vector_size };
        if width == 0 || out.ty.vector_size() != width || out.elem_type() != op.list.elem_type() {
            return Err(invalid("uniform load result does not match its pointee"));
        }
        let ty = Scalar::memory_element(out.ty)?;
        self.half_target(ty)?;
        let index_type = Scalar::of(op.index.ty)?;
        if !matches!(index_type, Scalar::U32 | Scalar::U64) {
            return Err(invalid("uniform load index must be unsigned"));
        }
        let base = self.shared_array(op.list)?;
        let index = self.value(op.index)?;
        let address = self.reg(Scalar::U64);
        let stride = width.checked_mul(ty.bytes())
            .filter(|stride| *stride <= u32::MAX as usize)
            .ok_or_else(|| unsupported("uniform load stride exceeds U32"))?;
        let multiply = if index_type == Scalar::U32 { "mul.wide.u32" } else { "mul.lo.u64" };
        self.line(format!("{multiply} {address}, {index}, {stride};"));
        self.line(format!("add.u64 {address}, {base}, {address};"));
        self.line("bar.sync 0;");
        if is_atomic(op.list) {
            let pointer = Variable::new(out.kind, op.list.ty.with_vector_size(width));
            let destination = self.atomic_pointer_destination(pointer)?;
            self.line(format!("cvta.shared.u64 {destination}, {address};"));
            self.atomic(&AtomicOp::Load(ruda_core::ir::UnaryOperator { input: pointer }), out)
        } else {
            let registers = self.vector_destination(out)?;
            for (lane, register) in registers.iter().enumerate() {
                let element_address = format!("{address}+{}", lane * ty.bytes());
                self.load_memory_scalar("shared", ty, register, &element_address);
            }
            Ok(())
        }
    }

    pub fn atomic_instruction(&mut self, instruction: &Instruction) -> Result<bool> {
        let output = || instruction.out.ok_or_else(|| invalid("atomic instruction requires output"));
        match &instruction.operation {
            Operation::Atomic(op) => self.atomic(op, output()?)?,
            Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op)) if is_atomic(op.list) => {
                if (op.vector_size > 1 && op.vector_size != op.list.ty.vector_size()) || op.unroll_factor != 1 {
                    return Err(unsupported("vectorized atomic pointer indexing"));
                }
                let checked = matches!(&instruction.operation, Operation::Operator(Operator::Index(_)));
                self.atomic_index(op.list, op.index, output()?, checked)?;
            }
            Operation::Operator(Operator::Reinterpret(op)) if instruction.out.is_some_and(is_atomic) => {
                let out = output()?;
                if Scalar::of(op.input.ty)? != Scalar::U64 {
                    return Err(invalid("native atomic pointer must be reinterpreted from a u64 address"));
                }
                let address = self.value(op.input)?;
                let destination = self.atomic_pointer_destination(out)?;
                self.line(format!("mov.u64 {destination}, {address};"));
                self.native_atomic_pointers.insert(destination);
            }
            Operation::Copy(input) if is_atomic(*input) => {
                let out = output()?;
                if out.ty != input.ty { return Err(invalid("atomic pointer copy type mismatch")); }
                let source = self.atomic_pointer(*input)?;
                let destination = self.atomic_pointer_destination(out)?;
                self.line(format!("mov.u64 {destination}, {source};"));
                if self.native_atomic_pointers.contains(&source) {
                    self.native_atomic_pointers.insert(destination);
                }
            }
            Operation::Operator(Operator::Select(op)) if instruction.out.is_some_and(is_atomic) => {
                let out = output()?;
                if out.ty != op.then.ty || out.ty != op.or_else.ty || Scalar::of(op.cond.ty)? != Scalar::Pred {
                    return Err(invalid("atomic pointer select operand types"));
                }
                let condition = self.value(op.cond)?;
                let yes = self.atomic_pointer(op.then)?;
                let no = self.atomic_pointer(op.or_else)?;
                let destination = self.atomic_pointer_destination(out)?;
                self.line(format!("selp.u64 {destination}, {yes}, {no}, {condition};"));
                if self.native_atomic_pointers.contains(&yes) || self.native_atomic_pointers.contains(&no) {
                    self.native_atomic_pointers.insert(destination);
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn atomic(&mut self, operation: &AtomicOp, out: Variable) -> Result<()> {
        let reflected = Operation::Atomic(operation.clone());
        let args = reflected.args().ok_or_else(|| invalid("atomic operands missing"))?;
        let width = out.ty.vector_size();
        if width > 1 {
            if args.iter().any(|arg| arg.ty.vector_size() != width) {
                return Err(invalid("atomic vector operand widths differ"));
            }
            if self.atomic_vector(operation, out)? {
                return Ok(());
            }
            let operands = args.iter().map(|arg| {
                if is_atomic(*arg) { Ok(Vec::new()) } else { self.vector_values(*arg) }
            }).collect::<Result<Vec<_>>>()?;
            let destinations = if is_atomic(out) { Vec::new() } else { self.vector_destination(out)? };
            for lane in 0..width {
                let mut pointers = Vec::new();
                let mut registers = Vec::new();
                for (index, variable) in args.iter().chain(std::iter::once(&out)).enumerate() {
                    let scalar = Variable::new(variable.kind, variable.ty.with_vector_size(1));
                    if is_atomic(*variable) {
                        let base = self.atomic_pointer(*variable)?;
                        let pointer = self.reg(Scalar::U64);
                        self.line(format!("add.u64 {pointer}, {base}, {};", lane * element(*variable)?.bytes()));
                        if self.native_atomic_pointers.contains(&base) {
                            self.native_atomic_pointers.insert(pointer.clone());
                        }
                        pointers.push((scalar, self.atomic_pointers.insert(scalar, pointer)));
                    } else {
                        let register = if index == args.len() { &destinations[lane] } else { &operands[index][lane] };
                        registers.push((scalar, self.registers.insert(scalar, register.clone())));
                    }
                }
                let scalar_args = args.iter().map(|arg| Variable::new(arg.kind, arg.ty.with_vector_size(1))).collect::<Vec<_>>();
                let scalar_op = Operation::from_code_and_args(reflected.op_code(), &scalar_args)
                    .ok_or_else(|| invalid("atomic vector reflection failed"))?;
                let result = self.instruction(Instruction::new(scalar_op, Variable::new(out.kind, out.ty.with_vector_size(1))));
                for (variable, previous) in pointers.into_iter().rev() {
                    if let Some(previous) = previous { self.atomic_pointers.insert(variable, previous); }
                    else { self.atomic_pointers.remove(&variable); }
                }
                for (variable, previous) in registers.into_iter().rev() {
                    if let Some(previous) = previous { self.registers.insert(variable, previous); }
                    else { self.registers.remove(&variable); }
                }
                result?;
            }
            return Ok(());
        }
        match operation {
            AtomicOp::Load(op) => {
                let ty = element(op.input)?;
                if Scalar::of(out.ty)? != ty { return Err(invalid("atomic load element type mismatch")); }
                let pointer = self.atomic_pointer(op.input)?;
                let destination = self.destination(out)?;
                if ty.narrow() {
                    return self.atomic_integer_compare_exchange(ty, &pointer, "0", "0", &destination);
                }
                self.atomic_target(ty, "cas")?;
                let semantics = self.atomic_semantics(&pointer);
                self.line(format!("atom{semantics}.cas.{} {destination}, [{pointer}], 0, 0;", ty.bits()));
                Ok(())
            }
            AtomicOp::Store(op) => {
                let ty = element(out)?;
                if Scalar::of(op.input.ty)? != ty { return Err(invalid("atomic store element type mismatch")); }
                let pointer = self.atomic_pointer(out)?;
                let value = self.value(op.input)?;
                let previous = self.reg(ty);
                self.atomic_reduce(ty, "exch", &pointer, &value, &previous)
            }
            AtomicOp::CompareAndSwap(op) => {
                let ty = element(op.input)?;
                if [op.cmp.ty, op.val.ty, out.ty].iter().any(|value| Scalar::of(*value).ok() != Some(ty)) {
                    return Err(invalid("atomic compare-and-swap element type mismatch"));
                }
                self.atomic_target(ty, "cas")?;
                let pointer = self.atomic_pointer(op.input)?;
                let compare = self.value(op.cmp)?;
                let value = self.value(op.val)?;
                let destination = self.destination(out)?;
                if ty.narrow() {
                    return self.atomic_integer_compare_exchange(ty, &pointer, &compare, &value, &destination);
                }
                let semantics = self.atomic_semantics(&pointer);
                self.line(format!("atom{semantics}.cas.{} {destination}, [{pointer}], {compare}, {value};", ty.bits()));
                Ok(())
            }
            AtomicOp::Swap(op) | AtomicOp::Add(op) | AtomicOp::Sub(op) |
            AtomicOp::Min(op) | AtomicOp::Max(op) | AtomicOp::And(op) |
            AtomicOp::Or(op) | AtomicOp::Xor(op) => {
                let ty = element(op.lhs)?;
                if Scalar::of(op.rhs.ty)? != ty || Scalar::of(out.ty)? != ty {
                    return Err(invalid("atomic reduction element type mismatch"));
                }
                let opcode = match operation {
                    AtomicOp::Swap(_) => "exch",
                    AtomicOp::Add(_) | AtomicOp::Sub(_) => "add",
                    AtomicOp::Min(_) => "min",
                    AtomicOp::Max(_) => "max",
                    AtomicOp::And(_) => "and",
                    AtomicOp::Or(_) => "or",
                    AtomicOp::Xor(_) => "xor",
                    _ => unreachable!(),
                };
                let pointer = self.atomic_pointer(op.lhs)?;
                let mut value = self.value(op.rhs)?;
                if matches!(operation, AtomicOp::Sub(_)) {
                    let negated = self.reg(ty);
                    if ty.integer() {
                        let suffix = if ty.bytes() == 8 { "u64" } else { "u32" };
                        self.line(format!("sub.{suffix} {negated}, 0, {value};"));
                    } else {
                        let mask = match ty.bytes() {
                            2 => "0x8000",
                            4 => "0x80000000",
                            _ => "0x8000000000000000",
                        };
                        self.line(format!("xor.{} {negated}, {value}, {mask};", ty.bits()));
                    }
                    value = negated;
                }
                let destination = self.destination(out)?;
                self.atomic_reduce(ty, opcode, &pointer, &value, &destination)
            }
        }
    }
}
