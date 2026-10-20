use super::*;
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use hashbrown::HashMap;

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
pub enum PointerOp {
    Address(TypedPointer),
    Offset(BinaryOperator),
    Component(BinaryOperator),
    ReadOnly(UnaryOperator),
    Select(Select),
    Load {
        pointer: Variable,
        checked: bool,
    },
    Store {
        pointer: Variable,
        value: Variable,
        checked: bool,
    },
    Atomic {
        pointer: Variable,
        #[cfg_attr(feature = "ir-serde", serde(with = "atomic_code"))]
        operation: AtomicOpCode,
        values: Vec<Variable>,
    },
    NativeAddress(UnaryOperator),
    FromNative {
        address: Variable,
        ty: PointerValueType,
    },
    UniformLoad {
        pointer: Variable,
    },
    ReferenceLoad {
        reference: Variable,
    },
    ReferenceStore {
        reference: Variable,
        value: Variable,
    },
    EraseSpace(UnaryOperator),
    Absent,
}

#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PointerOpCode {
    Address(PointerType, bool, usize),
    Offset,
    Component,
    ReadOnly,
    Select,
    Load(bool),
    Store(bool),
    Atomic(AtomicOpCode),
    NativeAddress,
    FromNative(PointerValueType),
    UniformLoad,
    ReferenceLoad,
    ReferenceStore,
    EraseSpace,
    Absent,
}

#[cfg(feature = "ir-serde")]
mod atomic_code {
    use super::AtomicOpCode;
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(op: &AtomicOpCode, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(match op {
            AtomicOpCode::Load => 0,
            AtomicOpCode::Store => 1,
            AtomicOpCode::Swap => 2,
            AtomicOpCode::Add => 3,
            AtomicOpCode::Sub => 4,
            AtomicOpCode::Min => 5,
            AtomicOpCode::Max => 6,
            AtomicOpCode::And => 7,
            AtomicOpCode::Or => 8,
            AtomicOpCode::Xor => 9,
            AtomicOpCode::CompareAndSwap => 10,
        })
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<AtomicOpCode, D::Error> {
        Ok(match u8::deserialize(deserializer)? {
            0 => AtomicOpCode::Load,
            1 => AtomicOpCode::Store,
            2 => AtomicOpCode::Swap,
            3 => AtomicOpCode::Add,
            4 => AtomicOpCode::Sub,
            5 => AtomicOpCode::Min,
            6 => AtomicOpCode::Max,
            7 => AtomicOpCode::And,
            8 => AtomicOpCode::Or,
            9 => AtomicOpCode::Xor,
            10 => AtomicOpCode::CompareAndSwap,
            _ => return Err(serde::de::Error::custom("invalid pointer atomic opcode")),
        })
    }
}

impl OperationReflect for PointerOp {
    type OpCode = PointerOpCode;
    fn is_pure(&self) -> bool {
        matches!(
            self,
            Self::Address(_)
                | Self::Offset(_)
                | Self::Component(_)
                | Self::ReadOnly(_)
                | Self::Select(_)
                | Self::NativeAddress(_)
                | Self::FromNative { .. }
                | Self::EraseSpace(_)
                | Self::Absent
        )
    }
    fn op_code(&self) -> Self::OpCode {
        match self {
            Self::Address(p) => PointerOpCode::Address(p.ty, p.checked, p.unroll_factor),
            Self::Offset(_) => PointerOpCode::Offset,
            Self::Component(_) => PointerOpCode::Component,
            Self::ReadOnly(_) => PointerOpCode::ReadOnly,
            Self::Select(_) => PointerOpCode::Select,
            Self::Load { checked, .. } => PointerOpCode::Load(*checked),
            Self::Store { checked, .. } => PointerOpCode::Store(*checked),
            Self::Atomic { operation, .. } => PointerOpCode::Atomic(*operation),
            Self::NativeAddress(_) => PointerOpCode::NativeAddress,
            Self::FromNative { ty, .. } => PointerOpCode::FromNative(*ty),
            Self::UniformLoad { .. } => PointerOpCode::UniformLoad,
            Self::ReferenceLoad { .. } => PointerOpCode::ReferenceLoad,
            Self::ReferenceStore { .. } => PointerOpCode::ReferenceStore,
            Self::EraseSpace(_) => PointerOpCode::EraseSpace,
            Self::Absent => PointerOpCode::Absent,
        }
    }
    fn args(&self) -> Option<Vec<Variable>> {
        Some(match self {
            Self::Address(p) => vec![p.base, p.index],
            Self::Offset(p) | Self::Component(p) => vec![p.lhs, p.rhs],
            Self::ReadOnly(p) | Self::NativeAddress(p) => vec![p.input],
            Self::Select(p) => vec![p.cond, p.then, p.or_else],
            Self::Load { pointer, .. } | Self::UniformLoad { pointer } => vec![*pointer],
            Self::Store { pointer, value, .. } => vec![*pointer, *value],
            Self::Atomic {
                pointer, values, ..
            } => {
                let mut args = vec![*pointer];
                args.extend(values);
                args
            }
            Self::FromNative { address, .. } => vec![*address],
            Self::ReferenceLoad { reference } => vec![*reference],
            Self::ReferenceStore { reference, value } => vec![*reference, *value],
            Self::EraseSpace(op) => vec![op.input],
            Self::Absent => vec![],
        })
    }
    fn from_code_and_args(code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        Some(match (code, args) {
            (PointerOpCode::Address(ty, checked, unroll_factor), [base, index]) => {
                Self::Address(TypedPointer {
                    base: *base,
                    index: *index,
                    ty,
                    checked,
                    unroll_factor,
                })
            }
            (PointerOpCode::Offset, [lhs, rhs]) => Self::Offset(BinaryOperator {
                lhs: *lhs,
                rhs: *rhs,
            }),
            (PointerOpCode::Component, [lhs, rhs]) => Self::Component(BinaryOperator {
                lhs: *lhs, rhs: *rhs,
            }),
            (PointerOpCode::ReadOnly, [input]) => Self::ReadOnly(UnaryOperator { input: *input }),
            (PointerOpCode::Select, [cond, then, or_else]) => Self::Select(Select {
                cond: *cond,
                then: *then,
                or_else: *or_else,
            }),
            (PointerOpCode::Load(checked), [pointer]) => Self::Load {
                pointer: *pointer,
                checked,
            },
            (PointerOpCode::UniformLoad, [pointer]) => Self::UniformLoad {
                pointer: *pointer,
            },
            (PointerOpCode::Store(checked), [pointer, value]) => Self::Store {
                pointer: *pointer,
                value: *value,
                checked,
            },
            (PointerOpCode::Atomic(operation), [pointer, values @ ..]) => Self::Atomic {
                pointer: *pointer,
                operation,
                values: values.to_vec(),
            },
            (PointerOpCode::NativeAddress, [input]) => {
                Self::NativeAddress(UnaryOperator { input: *input })
            }
            (PointerOpCode::FromNative(ty), [address]) => Self::FromNative {
                address: *address,
                ty,
            },
            (PointerOpCode::ReferenceLoad, [reference]) => Self::ReferenceLoad { reference: *reference },
            (PointerOpCode::ReferenceStore, [reference, value]) => Self::ReferenceStore {
                reference: *reference, value: *value,
            },
            (PointerOpCode::EraseSpace, [input]) => Self::EraseSpace(UnaryOperator { input: *input }),
            (PointerOpCode::Absent, []) => Self::Absent,
            _ => return None,
        })
    }
}

fn pointer_ty(value: Variable) -> Result<PointerValueType, String> {
    value
        .ty
        .pointer_type()
        .ok_or_else(|| format!("expected pointer value, got {}", value.ty))
}

impl PointerOp {
    pub fn validate(&self, out: Option<Variable>) -> Result<(), String> {
        let output = || out.ok_or_else(|| String::from("pointer operation requires an output"));
        let same = |left: Type, right: Type| {
            if left == right {
                Ok(())
            } else {
                Err(String::from("pointer operation type mismatch"))
            }
        };
        match self {
            Self::Absent => {
                let ty = pointer_ty(output()?)?;
                if ty.width == 0 { return Err("absent reference requires a sized pointee".into()); }
                Ok(())
            }
            Self::EraseSpace(op) => {
                let mut ty = pointer_ty(op.input)?;
                ty.address_space = AddressSpace::Generic;
                same(output()?.ty, Type::Pointer(ty))
            }
            Self::ReferenceLoad { .. } | Self::ReferenceStore { .. } => {
                Err("reference binding operations must be lowered before pointer validation".into())
            }
            Self::Address(p) => {
                p.validate()?;
                same(output()?.ty, p.ty.value_type())
            }
            Self::Offset(op) => {
                pointer_ty(op.lhs)?;
                same(output()?.ty, op.lhs.ty)?;
                if !op.rhs.ty.is_unsigned_int() || op.rhs.ty.vector_size() != 1 {
                    return Err("pointer offset must be a scalar unsigned integer".into());
                }
                Ok(())
            }
            Self::Component(op) => {
                let source = pointer_ty(op.lhs)?;
                let destination = pointer_ty(output()?)?;
                let mut scalar = source;
                scalar.width = 1;
                if destination.access == PointerAccess::ReadOnly { scalar.access = PointerAccess::ReadOnly; }
                if source.width == 0 || scalar != destination
                    || !op.rhs.ty.is_unsigned_int() || op.rhs.ty.vector_size() != 1
                { return Err("pointer component requires a scalar index and compatible scalar view".into()); }
                Ok(())
            }
            Self::ReadOnly(op) => {
                let mut ty = pointer_ty(op.input)?;
                ty.access = PointerAccess::ReadOnly;
                same(output()?.ty, Type::Pointer(ty))
            }
            Self::Select(op) => {
                pointer_ty(op.then)?;
                same(op.then.ty, op.or_else.ty)?;
                same(output()?.ty, op.then.ty)?;
                same(op.cond.ty, Type::scalar(ElemType::Bool))
            }
            Self::Load { pointer, .. } => {
                let ty = pointer_ty(*pointer)?;
                if ty.storage.is_atomic() {
                    return Err("ordinary pointer load requires a non-atomic pointee".into());
                }
                same(output()?.ty, ty.pointee())
            }
            Self::UniformLoad { pointer } => {
                let ty = pointer_ty(*pointer)?;
                if ty.address_space != AddressSpace::Shared || ty.width == 0 {
                    return Err("workgroup uniform load requires a shared-memory pointer".into());
                }
                let result = match ty.storage {
                    StorageType::Scalar(_) | StorageType::Packed(_, _) => ty.pointee(),
                    StorageType::Atomic(elem) => Type::scalar(elem).with_vector_size(ty.width),
                    _ => return Err("workgroup uniform load requires a numeric or boolean pointee".into()),
                };
                same(output()?.ty, result)
            }
            Self::Store { pointer, value, .. } => {
                let ty = pointer_ty(*pointer)?;
                if ty.access != PointerAccess::ReadWrite || ty.storage.is_atomic() || out.is_some()
                {
                    return Err("ordinary pointer store requires a writable non-atomic pointer and no output".into());
                }
                same(value.ty, ty.pointee())
            }
            Self::Atomic {
                pointer,
                operation,
                values,
            } => {
                let ty = pointer_ty(*pointer)?;
                let StorageType::Atomic(elem) = ty.storage else {
                    return Err("typed atomic requires an atomic pointee".into());
                };
                if ty.width == 0 {
                    return Err("typed atomic requires nonzero storage width".into());
                }
                if *operation != AtomicOpCode::Load && ty.access != PointerAccess::ReadWrite {
                    return Err("atomic mutation requires a writable pointer".into());
                }
                let count = match operation {
                    AtomicOpCode::Load => 0,
                    AtomicOpCode::CompareAndSwap => 2,
                    _ => 1,
                };
                if values.len() != count
                    || values
                        .iter()
                        .any(|value| value.ty != Type::scalar(elem).with_vector_size(ty.width))
                {
                    return Err("typed atomic operand mismatch".into());
                }
                if *operation == AtomicOpCode::Store {
                    if out.is_some() {
                        return Err("atomic store has no value result".into());
                    }
                    Ok(())
                } else {
                    same(output()?.ty, Type::scalar(elem).with_vector_size(ty.width))
                }
            }
            Self::NativeAddress(op) => {
                pointer_ty(op.input)?;
                same(output()?.ty, Type::scalar(ElemType::UInt(UIntKind::U64)))
            }
            Self::FromNative { address, ty } => {
                if ty.width == 0 {
                    return Err("native pointer import requires nonzero pointee width".into());
                }
                same(address.ty, Type::scalar(ElemType::UInt(UIntKind::U64)))?;
                same(output()?.ty, Type::Pointer(*ty))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Storage(Variable, usize),
    Native,
    Absent,
}

#[derive(Clone, Copy)]
struct Handle {
    origin: Variable,
    index: Variable,
    native: Option<Variable>,
}

/// Lower pointer values to the logical storage ABI, either before or after SSA.
pub struct PointerValueLowering {
    origins: Vec<Origin>,
    reachable: HashMap<Variable, Vec<usize>>,
    handles: HashMap<Variable, Handle>,
    index_ty: Type,
}

fn walk(
    scope: &Scope,
    visit: &mut impl FnMut(&Instruction) -> Result<(), String>,
) -> Result<(), String> {
    for inst in &scope.instructions {
        visit(inst)?;
        if let Operation::CoopMma(super::CoopMma::Elementwise { function, .. }) = &inst.operation {
            walk(&function.body, visit)?;
        }
        if let Operation::Branch(branch) = &inst.operation {
            match branch {
                Branch::If(b) => walk(&b.scope, visit)?,
                Branch::IfElse(b) => {
                    walk(&b.scope_if, visit)?;
                    walk(&b.scope_else, visit)?;
                }
                Branch::Switch(b) => {
                    walk(&b.scope_default, visit)?;
                    for (_, scope) in &b.cases {
                        walk(scope, visit)?;
                    }
                }
                Branch::RangeLoop(b) => walk(&b.scope, visit)?,
                Branch::Loop(b) => walk(&b.scope, visit)?,
                _ => {}
            }
        }
    }
    Ok(())
}

impl PointerValueLowering {
    /// Validate pointer definitions and collect their reachable storage origins.
    pub fn new(scope: &Scope) -> Result<Self, String> {
        let index_ty = Type::new(
            scope
                .resolve_type::<usize>()
                .unwrap_or(UIntKind::U32.into()),
        );
        let mut lowering = Self {
            origins: vec![],
            reachable: HashMap::new(),
            handles: HashMap::new(),
            index_ty,
        };
        let mut edges = Vec::new();
        walk(scope, &mut |inst| {
            if let Operation::CoopMma(super::CoopMma::Elementwise { function, captures, .. }) = &inst.operation {
                for (formal, actual) in function.body.parameters[function.num_arguments..]
                    .iter().zip(captures)
                {
                    if formal.ty.is_pointer() {
                        if formal.ty != actual.ty {
                            return Err("closure pointer capture type mismatch".into());
                        }
                        edges.push((*formal, *actual));
                    }
                }
            } else if let Operation::Pointer(op) = &inst.operation {
                op.validate(inst.out)?;
                match op {
                    PointerOp::Absent => {
                        let id = lowering.origin_id(Origin::Absent);
                        lowering.reachable.entry(inst.out()).or_default().push(id);
                    }
                    PointerOp::Address(p) => {
                        if p.index.ty.size() > lowering.index_ty.size() {
                            lowering.index_ty = p.index.ty;
                        }
                        let origin = Origin::Storage(p.base, p.unroll_factor);
                        let id = lowering.origin_id(origin);
                        lowering.reachable.entry(inst.out()).or_default().push(id);
                    }
                    PointerOp::FromNative { .. } => {
                        let id = lowering.origin_id(Origin::Native);
                        lowering.reachable.entry(inst.out()).or_default().push(id);
                    }
                    PointerOp::Offset(p) | PointerOp::Component(p) => {
                        if p.rhs.ty.size() > lowering.index_ty.size() {
                            lowering.index_ty = p.rhs.ty;
                        }
                        edges.push((inst.out(), p.lhs));
                    }
                    PointerOp::ReadOnly(p) | PointerOp::EraseSpace(p) => edges.push((inst.out(), p.input)),
                    PointerOp::Select(p) => {
                        edges.push((inst.out(), p.then));
                        edges.push((inst.out(), p.or_else));
                    }
                    _ => {}
                }
            } else if inst.out.is_some_and(|out| out.ty.is_pointer()) {
                match inst.operation {
                    Operation::Copy(input) if inst.out().ty == input.ty => {
                        edges.push((inst.out(), input))
                    }
                    _ => return Err("pointer values cannot be used by numeric operations".into()),
                }
            } else if inst
                .operation
                .args()
                .unwrap_or_default()
                .iter()
                .any(|value| value.ty.is_pointer())
            {
                return Err("pointer operands require pointer operations".into());
            }
            Ok(())
        })?;
        // A monotone union reaches a fixed point even for loop-carried pointer assignments.
        loop {
            let mut changed = false;
            for (out, input) in &edges {
                let roots = lowering.reachable.get(input).cloned().unwrap_or_default();
                let dest = lowering.reachable.entry(*out).or_default();
                for root in roots {
                    if !dest.contains(&root) {
                        dest.push(root);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        walk(scope, &mut |inst| {
            let args = match &inst.operation {
                Operation::Pointer(op) => op.args().unwrap_or_default(),
                Operation::Copy(input) if input.ty.is_pointer() => vec![*input],
                Operation::CoopMma(super::CoopMma::Elementwise { captures, .. }) => captures.clone(),
                _ => vec![],
            };
            for var in args.into_iter().filter(|v| v.ty.is_pointer()) {
                if lowering.reachable.get(&var).is_none_or(Vec::is_empty) {
                    return Err("pointer has no storage origin".into());
                }
            }
            Ok(())
        })?;
        Ok(lowering)
    }

    fn origin_id(&mut self, origin: Origin) -> usize {
        if let Some(id) = self.origins.iter().position(|old| *old == origin) {
            id
        } else {
            self.origins.push(origin);
            self.origins.len() - 1
        }
    }

    fn handle(&mut self, scope: &Scope, var: Variable) -> Handle {
        *self.handles.entry(var).or_insert_with(|| {
            let allocate = |ty| {
                let kind = match var.kind {
                    VariableKind::LocalMut { .. } => VariableKind::LocalMut {
                        id: scope.allocator.new_local_index(),
                    },
                    VariableKind::LocalConst { .. } => VariableKind::LocalConst {
                        id: scope.allocator.new_local_index(),
                    },
                    VariableKind::Versioned { .. } => VariableKind::LocalConst {
                        id: scope.allocator.new_local_index(),
                    },
                    _ => panic!("pointer value must be local"),
                };
                Variable::new(kind, ty)
            };
            let native = self
                .reachable
                .get(&var)
                .is_some_and(|roots| {
                    roots
                        .iter()
                        .any(|root| self.origins[*root] == Origin::Native)
                })
                .then(|| allocate(UIntKind::U64.into()));
            Handle {
                origin: allocate(UIntKind::U32.into()),
                index: allocate(self.index_ty),
                native,
            }
        })
    }

    /// Return the storage selector, element offset, and optional native address for a value.
    pub fn components(&mut self, scope: &Scope, var: Variable) -> Vec<Variable> {
        let handle = self.handle(scope, var);
        let mut components = vec![handle.origin, handle.index];
        components.extend(handle.native);
        components
    }

    fn cast(scope: &mut Scope, input: Variable, ty: Type) -> Variable {
        if input.ty == ty {
            input
        } else {
            let out = *scope.create_local(ty);
            scope.register(Instruction::new(
                Operator::Cast(UnaryOperator { input }),
                out,
            ));
            out
        }
    }

    fn copy(scope: &mut Scope, dest: Handle, source: Handle) {
        scope.register(Instruction::new(
            Operation::Copy(source.origin),
            dest.origin,
        ));
        scope.register(Instruction::new(Operation::Copy(source.index), dest.index));
        if let Some(native) = dest.native {
            scope.register(Instruction::new(
                Operation::Copy(source.native.unwrap_or_else(|| 0u64.into())),
                native,
            ));
        }
    }

    fn access(
        &self,
        scope: &mut Scope,
        origin: Origin,
        handle: Handle,
        pointer: Variable,
        op: &PointerOp,
        out: Option<Variable>,
    ) -> Result<(), String> {
        let ty = pointer_ty(pointer)?;
        let checked = match op {
            PointerOp::Load { checked, .. } | PointerOp::Store { checked, .. } => *checked,
            _ => true,
        };
        match origin {
            Origin::Absent => {
                scope.register(Instruction::no_out(Branch::Unreachable));
            }
            Origin::Storage(base, unroll_factor) => {
                let mut view = ty.view();
                if view.address_space == AddressSpace::Generic {
                    view.address_space = match base.kind {
                        VariableKind::GlobalInputArray(_) | VariableKind::GlobalOutputArray(_) => AddressSpace::Global,
                        VariableKind::SharedArray { .. } => AddressSpace::Shared,
                        VariableKind::LocalArray { .. } | VariableKind::ConstantArray { .. } => AddressSpace::Local,
                        _ => return Err("reference origin is not addressable storage".into()),
                    };
                }
                let ptr = TypedPointer {
                    base,
                    index: handle.index,
                    ty: view,
                    checked,
                    unroll_factor,
                };
                match op {
                    PointerOp::Load { .. } => {
                        scope.register(Instruction::new(TypedMemory::Load(ptr), out.unwrap()));
                    }
                    PointerOp::UniformLoad { .. } => {
                        scope.register(Instruction::new(
                            Operator::WorkgroupUniformLoad(IndexOperator {
                                list: base,
                                index: handle.index,
                                vector_size: if ty.width == base.ty.vector_size() { 0 } else { ty.width },
                                unroll_factor,
                            }),
                            out.unwrap(),
                        ));
                    }
                    PointerOp::Store { value, .. } => {
                        scope.register(Instruction::new(
                            TypedMemory::Store {
                                pointer: ptr,
                                value: *value,
                            },
                            base,
                        ));
                    }
                    PointerOp::NativeAddress(_) => {
                        let address = *scope.create_local(UIntKind::U64.into());
                        scope.register(Instruction::new(
                            Operator::NativeAddress(BinaryOperator {
                                lhs: base,
                                rhs: handle.index.ty.constant(ConstantValue::UInt(0)),
                            }),
                            address,
                        ));
                        let index = Self::cast(scope, handle.index, UIntKind::U64.into());
                        let bytes = *scope.create_local(UIntKind::U64.into());
                        scope.register(Instruction::new(
                            Arithmetic::Mul(BinaryOperator {
                                lhs: index,
                                rhs: (ty.pointee().size() as u64).into(),
                            }),
                            bytes,
                        ));
                        scope.register(Instruction::new(
                            Arithmetic::Add(BinaryOperator {
                                lhs: address,
                                rhs: bytes,
                            }),
                            out.unwrap(),
                        ));
                    }
                    PointerOp::Atomic {
                        operation, values, ..
                    } => {
                        let atomic = *scope.create_local(ty.pointee());
                        scope.register(Instruction::new(
                            Operator::UncheckedIndex(IndexOperator {
                                list: base,
                                index: handle.index,
                                vector_size: if ty.width == base.ty.vector_size() {
                                    0
                                } else {
                                    ty.width
                                },
                                unroll_factor,
                            }),
                            atomic,
                        ));
                        let args = if *operation == AtomicOpCode::Store {
                            values.clone()
                        } else {
                            let mut args = vec![atomic];
                            args.extend(values);
                            args
                        };
                        let operation = AtomicOp::from_code_and_args(*operation, &args)
                            .ok_or_else(|| String::from("invalid typed atomic opcode"))?;
                        scope.register(Instruction::new(operation, out.unwrap_or(atomic)));
                    }
                    _ => unreachable!(),
                }
            }
            Origin::Native => match op {
                PointerOp::UniformLoad { .. } => {
                    scope.register(Synchronization::SyncRuda);
                    if ty.storage.is_atomic() {
                        let atomic = *scope.create_local(ty.pointee());
                        scope.register(Instruction::new(
                            Operator::Reinterpret(UnaryOperator { input: handle.native.unwrap() }),
                            atomic,
                        ));
                        scope.register(Instruction::new(
                            AtomicOp::Load(UnaryOperator { input: atomic }),
                            out.unwrap(),
                        ));
                    } else {
                        scope.register(Instruction::new(
                            Operator::NativeLoad(UnaryOperator { input: handle.native.unwrap() }),
                            out.unwrap(),
                        ));
                    }
                }
                PointerOp::Load { .. } => {
                    scope.register(Instruction::new(
                        Operator::NativeLoad(UnaryOperator {
                            input: handle.native.unwrap(),
                        }),
                        out.unwrap(),
                    ));
                }
                PointerOp::Store { value, .. } => {
                    scope.register(Instruction::no_out(Operator::NativeStore(BinaryOperator {
                        lhs: handle.native.unwrap(),
                        rhs: *value,
                    })));
                }
                PointerOp::NativeAddress(_) => {
                    scope.register(Instruction::new(
                        Operation::Copy(handle.native.unwrap()),
                        out.unwrap(),
                    ));
                }
                PointerOp::Atomic {
                    operation, values, ..
                } => {
                    let atomic = *scope.create_local(ty.pointee());
                    scope.register(Instruction::new(
                        Operator::Reinterpret(UnaryOperator {
                            input: handle.native.unwrap(),
                        }),
                        atomic,
                    ));
                    let args = if *operation == AtomicOpCode::Store {
                        values.clone()
                    } else {
                        let mut args = vec![atomic];
                        args.extend(values);
                        args
                    };
                    let operation = AtomicOp::from_code_and_args(*operation, &args)
                        .ok_or_else(|| String::from("invalid native typed atomic opcode"))?;
                    scope.register(Instruction::new(operation, out.unwrap_or(atomic)));
                }
                _ => unreachable!(),
            },
        }
        Ok(())
    }

    /// Replace pointer operations recursively using the collected storage origins.
    pub fn lower_scope(&mut self, scope: &mut Scope) -> Result<(), String> {
        let original = core::mem::take(&mut scope.instructions);
        scope.locals.retain(|var| !var.ty.is_pointer());
        for inst in original {
            let start = scope.instructions.len();
            let source_loc = inst.source_loc.clone();
            let modes = inst.modes;
            match inst.operation {
                Operation::Pointer(ref op) => match op {
                    PointerOp::Absent => {
                        let handle = self.handle(scope, inst.out());
                        let origin = self.origin_id(Origin::Absent);
                        scope.register(Instruction::new(Operation::Copy((origin as u32).into()), handle.origin));
                        scope.register(Instruction::new(
                            Operation::Copy(self.index_ty.constant(0u32.into())), handle.index,
                        ));
                        if let Some(native) = handle.native {
                            scope.register(Instruction::new(Operation::Copy(0u64.into()), native));
                        }
                    }
                    PointerOp::ReferenceLoad { .. } | PointerOp::ReferenceStore { .. } => {
                        return Err("reference binding operations must be lowered before pointer values".into());
                    }
                    PointerOp::Address(p) => {
                        let handle = self.handle(scope, inst.out());
                        let id = self.origin_id(Origin::Storage(p.base, p.unroll_factor));
                        scope.register(Instruction::new(
                            Operation::Copy((id as u32).into()),
                            handle.origin,
                        ));
                        let index = Self::cast(scope, p.index, self.index_ty);
                        scope.register(Instruction::new(Operation::Copy(index), handle.index));
                        if let Some(native) = handle.native {
                            scope.register(Instruction::new(Operation::Copy(0u64.into()), native));
                        }
                    }
                    PointerOp::FromNative { address, .. } => {
                        let handle = self.handle(scope, inst.out());
                        let id = self.origin_id(Origin::Native);
                        scope.register(Instruction::new(
                            Operation::Copy((id as u32).into()),
                            handle.origin,
                        ));
                        scope.register(Instruction::new(
                            Operation::Copy(self.index_ty.constant(ConstantValue::UInt(0))),
                            handle.index,
                        ));
                        scope.register(Instruction::new(
                            Operation::Copy(*address),
                            handle.native.unwrap(),
                        ));
                    }
                    PointerOp::ReadOnly(p) | PointerOp::EraseSpace(p) => {
                        let dest = self.handle(scope, inst.out());
                        let source = self.handle(scope, p.input);
                        Self::copy(scope, dest, source);
                    }
                    PointerOp::Offset(p) => {
                        let source = self.handle(scope, p.lhs);
                        let dest = self.handle(scope, inst.out());
                        scope.register(Instruction::new(
                            Operation::Copy(source.origin),
                            dest.origin,
                        ));
                        let index = Self::cast(scope, p.rhs, self.index_ty);
                        scope.register(Instruction::new(
                            Arithmetic::Add(BinaryOperator {
                                lhs: source.index,
                                rhs: index,
                            }),
                            dest.index,
                        ));
                        if self.reachable[&p.lhs]
                            .iter()
                            .any(|root| self.origins[*root] == Origin::Native)
                        {
                            let delta = Self::cast(scope, p.rhs, UIntKind::U64.into());
                            let bytes = *scope.create_local(UIntKind::U64.into());
                            scope.register(Instruction::new(
                                Arithmetic::Mul(BinaryOperator {
                                    lhs: delta,
                                    rhs: (pointer_ty(p.lhs)?.pointee().size() as u64).into(),
                                }),
                                bytes,
                            ));
                            scope.register(Instruction::new(
                                Arithmetic::Add(BinaryOperator {
                                    lhs: source.native.unwrap(),
                                    rhs: bytes,
                                }),
                                dest.native.unwrap(),
                            ));
                        }
                    }
                    PointerOp::Component(p) => {
                        let source = self.handle(scope, p.lhs);
                        let dest = self.handle(scope, inst.out());
                        let width = pointer_ty(p.lhs)?.width;
                        let index = Self::cast(scope, p.rhs, self.index_ty);
                        scope.register(Instruction::new(Operation::Copy(source.origin), dest.origin));
                        let start = *scope.create_local(self.index_ty);
                        scope.register(Instruction::new(Arithmetic::Mul(BinaryOperator {
                            lhs: source.index, rhs: self.index_ty.constant((width as u64).into()),
                        }), start));
                        let offset = *scope.create_local(self.index_ty);
                        scope.register(Instruction::new(Arithmetic::Add(BinaryOperator { lhs: start, rhs: index }), offset));
                        let in_bounds = *scope.create_local(Type::scalar(ElemType::Bool));
                        scope.register(Instruction::new(Comparison::Lower(BinaryOperator {
                            lhs: index, rhs: self.index_ty.constant((width as u64).into()),
                        }), in_bounds));
                        let invalid = if self.index_ty.size() == 8 { u64::MAX } else { u32::MAX as u64 };
                        scope.register(Instruction::new(Operator::Select(Select {
                            cond: in_bounds, then: offset, or_else: self.index_ty.constant(invalid.into()),
                        }), dest.index));
                        if let Some(native) = dest.native {
                            let delta = Self::cast(scope, index, UIntKind::U64.into());
                            let bytes = *scope.create_local(UIntKind::U64.into());
                            scope.register(Instruction::new(Arithmetic::Mul(BinaryOperator {
                                lhs: delta, rhs: (pointer_ty(inst.out())?.pointee().size() as u64).into(),
                            }), bytes));
                            scope.register(Instruction::new(Arithmetic::Add(BinaryOperator {
                                lhs: source.native.unwrap(), rhs: bytes,
                            }), native));
                        }
                    }
                    PointerOp::Select(p) => {
                        let dest = self.handle(scope, inst.out());
                        let then = self.handle(scope, p.then);
                        let otherwise = self.handle(scope, p.or_else);
                        for (out, then, or_else) in [
                            (dest.origin, then.origin, otherwise.origin),
                            (dest.index, then.index, otherwise.index),
                        ] {
                            scope.register(Instruction::new(
                                Operator::Select(Select {
                                    cond: p.cond,
                                    then,
                                    or_else,
                                }),
                                out,
                            ));
                        }
                        if let Some(out) = dest.native {
                            scope.register(Instruction::new(
                                Operator::Select(Select {
                                    cond: p.cond,
                                    then: then.native.unwrap_or_else(|| 0u64.into()),
                                    or_else: otherwise.native.unwrap_or_else(|| 0u64.into()),
                                }),
                                out,
                            ));
                        }
                    }
                    PointerOp::Load { pointer, .. }
                    | PointerOp::UniformLoad { pointer }
                    | PointerOp::Store { pointer, .. }
                    | PointerOp::Atomic { pointer, .. }
                    | PointerOp::NativeAddress(UnaryOperator { input: pointer }) => {
                        let handle = self.handle(scope, *pointer);
                        let roots = self.reachable[pointer].clone();
                        if roots.len() == 1 {
                            if self.origins[roots[0]] == Origin::Absent {
                                scope.instructions.push(Instruction {
                                    operation: Branch::Unreachable.into(), out: None, ..inst.clone()
                                });
                                break;
                            }
                            self.access(
                                scope,
                                self.origins[roots[0]],
                                handle,
                                *pointer,
                                op,
                                inst.out,
                            )?;
                        } else {
                            let result = inst.out.map(|out| {
                                let value = *scope.create_local_restricted(out.ty);
                                scope.add_local_mut(value);
                                value
                            });
                            let mut cases = Vec::new();
                            for root in roots {
                                let mut branch = scope.child();
                                self.access(
                                    &mut branch,
                                    self.origins[root],
                                    handle,
                                    *pointer,
                                    op,
                                    result,
                                )?;
                                cases.push(((root as u32).into(), branch));
                            }
                            let mut scope_default = scope.child();
                            scope_default.register(Branch::Unreachable);
                            scope.register(Instruction::no_out(Branch::Switch(Box::new(Switch {
                                value: handle.origin,
                                scope_default,
                                cases,
                            }))));
                            if let Some(result) = result {
                                scope.register(Instruction::new(
                                    Operation::Copy(result),
                                    inst.out(),
                                ));
                            }
                        }
                    }
                },
                Operation::Copy(input) if input.ty.is_pointer() => {
                    let dest = self.handle(scope, inst.out());
                    let source = self.handle(scope, input);
                    Self::copy(scope, dest, source);
                }
                Operation::CoopMma(super::CoopMma::Elementwise {
                    input, mut function, captures,
                }) => {
                    let formals = function.body.parameters.clone();
                    let mut parameters = formals[..function.num_arguments].to_vec();
                    let mut lowered_captures = Vec::new();
                    for (formal, actual) in formals[function.num_arguments..]
                        .iter().copied().zip(captures)
                    {
                        if formal.ty.is_pointer() {
                            parameters.extend(self.components(&function.body, formal));
                            lowered_captures.extend(self.components(scope, actual));
                            for origin in self.reachable[&formal].clone() {
                                if let Origin::Storage(base, _) = self.origins[origin] {
                                    parameters.push(base);
                                    lowered_captures.push(base);
                                }
                            }
                        } else {
                            parameters.push(formal);
                            lowered_captures.push(actual);
                        }
                    }
                    function.body.parameters = parameters;
                    self.lower_scope(&mut function.body)?;
                    for var in function.body.allocator.take_variables() {
                        if !var.ty.is_pointer() {
                            function.body.add_local_mut(var);
                        }
                    }
                    for handle in self.handles.values() {
                        for var in [Some(handle.origin), Some(handle.index), handle.native].into_iter().flatten() {
                            if matches!(var.kind, VariableKind::LocalMut { .. }) {
                                function.body.add_local_mut(var);
                            }
                        }
                    }
                    scope.instructions.push(Instruction {
                        operation: Operation::CoopMma(super::CoopMma::Elementwise {
                            input, function, captures: lowered_captures,
                        }),
                        ..inst
                    });
                }
                Operation::Branch(mut branch) => {
                    match &mut branch {
                        Branch::If(b) => self.lower_scope(&mut b.scope)?,
                        Branch::IfElse(b) => {
                            self.lower_scope(&mut b.scope_if)?;
                            self.lower_scope(&mut b.scope_else)?;
                        }
                        Branch::Switch(b) => {
                            self.lower_scope(&mut b.scope_default)?;
                            for (_, scope) in &mut b.cases {
                                self.lower_scope(scope)?;
                            }
                        }
                        Branch::RangeLoop(b) => self.lower_scope(&mut b.scope)?,
                        Branch::Loop(b) => self.lower_scope(&mut b.scope)?,
                        _ => {}
                    }
                    scope.instructions.push(Instruction {
                        operation: Operation::Branch(branch),
                        ..inst
                    });
                }
                _ => scope.instructions.push(inst.clone()),
            }
            for emitted in &mut scope.instructions[start..] {
                emitted.source_loc = source_loc.clone();
                emitted.modes = modes;
            }
        }
        Ok(())
    }
}

impl Scope {
    /// Lower pointer values to storage selectors and element offsets before the existing SSA pass.
    /// The pointee/address-space/access descriptors remain on each resulting memory operation.
    pub fn lower_pointer_values(&mut self) -> Result<(), String> {
        self.lower_reference_values()?;
        let mut present = false;
        walk(self, &mut |inst| {
            present |= matches!(inst.operation, Operation::Pointer(_))
                || inst.out.is_some_and(|out| out.ty.is_pointer());
            Ok(())
        })?;
        if !present {
            return Ok(());
        }
        let mut lowering = PointerValueLowering::new(self)?;
        lowering.lower_scope(self)?;
        // Remove only replaced pointer declarations from the allocator's mutable pool.
        for var in self.allocator.take_variables() {
            if !var.ty.is_pointer() {
                self.add_local_mut(var);
            }
        }
        let mut variables = lowering
            .handles
            .values()
            .flat_map(|handle| [Some(handle.origin), Some(handle.index), handle.native])
            .flatten()
            .collect::<Vec<_>>();
        variables.sort_by_key(|var| match var.kind {
            VariableKind::LocalMut { id } | VariableKind::LocalConst { id } => id,
            _ => unreachable!(),
        });
        for var in variables {
            if matches!(var.kind, VariableKind::LocalMut { .. }) {
                self.add_local_mut(var);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> TypedPointer {
        let ty = Type::scalar(ElemType::Float(FloatKind::F32));
        TypedPointer {
            base: Variable::new(VariableKind::GlobalOutputArray(0), ty),
            index: 0u32.into(),
            ty: PointerType {
                pointee: ty,
                address_space: AddressSpace::Global,
                access: PointerAccess::ReadWrite,
            },
            checked: true,
            unroll_factor: 1,
        }
    }

    #[test]
    fn pointer_value_type_is_not_a_numeric_address() {
        let ty = view().ty.value_type();
        assert!(ty.is_pointer());
        assert!(!ty.is_unsigned_int() && !ty.is_atomic() && !ty.is_float());
        assert_eq!(ty.pointer_type().unwrap().pointee(), view().base.ty);
    }

    #[test]
    fn pointer_value_loop_lowering_retains_typed_storage() {
        let mut scope = Scope::root(false);
        let view = view();
        let pointer = *scope.create_local_restricted(view.ty.value_type());
        let next = *scope.create_local(view.ty.value_type());
        scope.add_local_mut(pointer);
        scope.register(Instruction::new(PointerOp::Address(view), pointer));
        let mut body = scope.child();
        body.register(Instruction::new(
            PointerOp::Offset(BinaryOperator {
                lhs: pointer,
                rhs: 1u32.into(),
            }),
            next,
        ));
        body.register(Instruction::new(Operation::Copy(next), pointer));
        body.register(Instruction::no_out(Branch::Break));
        scope.register(Instruction::no_out(Branch::Loop(Box::new(Loop {
            scope: body,
        }))));
        let out = *scope.create_local(view.ty.pointee);
        scope.register(Instruction::new(
            PointerOp::Load {
                pointer,
                checked: true,
            },
            out,
        ));
        scope.lower_pointer_values().unwrap();
        let mut loads = 0;
        walk(&scope, &mut |inst| {
            assert!(!matches!(inst.operation, Operation::Pointer(_)));
            assert!(!inst.out.is_some_and(|out| out.ty.is_pointer()));
            if let Operation::TypedMemory(TypedMemory::Load(ptr)) = &inst.operation {
                assert_eq!(ptr.base, view.base);
                assert_eq!(ptr.ty, view.ty);
                loads += 1;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(loads, 1);
    }

    #[test]
    fn pointer_value_rejects_permission_upgrade_and_numeric_copy() {
        let mut ty = view().ty;
        ty.access = PointerAccess::ReadOnly;
        let pointer = Variable::new(VariableKind::LocalConst { id: 3 }, ty.value_type());
        let writable = Variable::new(VariableKind::LocalConst { id: 4 }, view().ty.value_type());
        let mut scope = Scope::root(false);
        scope.register(Instruction::new(Operation::Copy(pointer), writable));
        assert!(scope.lower_pointer_values().is_err());
        let op = PointerOp::Store {
            pointer,
            value: 1.0f32.into(),
            checked: true,
        };
        assert!(op.validate(None).is_err());
        assert!(
            PointerOp::FromNative {
                address: 0u32.into(),
                ty: ty.value_type().pointer_type().unwrap()
            }
            .validate(Some(pointer))
            .is_err()
        );
    }

    #[cfg(feature = "ir-serde")]
    #[test]
    fn pointer_value_atomic_ir_roundtrips() {
        let mut ty = view().ty;
        ty.pointee = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
        let pointer = Variable::new(VariableKind::LocalConst { id: 3 }, ty.value_type());
        let op = PointerOp::Atomic {
            pointer,
            operation: AtomicOpCode::Add,
            values: vec![1u32.into()],
        };
        let encoded = serde_json::to_vec(&op).unwrap();
        assert_eq!(serde_json::from_slice::<PointerOp>(&encoded).unwrap(), op);
    }
}
