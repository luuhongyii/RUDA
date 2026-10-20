use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::hash::{Hash, Hasher};

use super::*;

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash, TypeHash)]
pub enum ReferenceKind {
    Memory(PointerValueType),
    Binding { target: Box<ReferenceKind>, writable: bool },
}

impl ReferenceKind {
    fn payload_type(&self) -> Type {
        match self {
            Self::Memory(ty) => Type::Pointer(*ty),
            Self::Binding { .. } => UIntKind::U32.into(),
        }
    }

    fn accepts(&self, source: &Self) -> bool {
        match (self, source) {
            (Self::Memory(destination), Self::Memory(source)) => {
                let mut source = *source;
                if destination.access == PointerAccess::ReadOnly {
                    source.access = PointerAccess::ReadOnly;
                }
                *destination == source
            }
            (Self::Binding { target: a, writable: aw }, Self::Binding { target: b, writable: bw }) => {
                a == b && (!*aw || *bw)
            }
            _ => false,
        }
    }
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash, TypeHash)]
pub enum ReferenceTarget {
    Memory(Variable),
    Binding { selector: Variable, target: Box<ReferenceKind>, writable: bool },
}

impl ReferenceTarget {
    fn kind(&self) -> ReferenceKind {
        match self {
            Self::Memory(pointer) => ReferenceKind::Memory(pointer.ty.pointer_type().expect("reference pointer")),
            Self::Binding { target, writable, .. } => ReferenceKind::Binding { target: target.clone(), writable: *writable },
        }
    }

    fn payload(&self) -> Variable {
        match self {
            Self::Memory(pointer) => *pointer,
            Self::Binding { selector, .. } => *selector,
        }
    }

    fn from_payload(kind: ReferenceKind, payload: Variable) -> Self {
        match kind {
            ReferenceKind::Memory(_) => Self::Memory(payload),
            ReferenceKind::Binding { target, writable } => Self::Binding { selector: payload, target, writable },
        }
    }
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, TypeHash)]
pub struct ReferenceData {
    bindings: Vec<(Variable, ReferenceTarget)>,
    locals: Vec<(Variable, Variable)>,
    #[cfg_attr(feature = "ir-serde", serde(default))]
    addressed: Vec<(Variable, bool)>,
    #[type_hash(skip)]
    #[cfg_attr(feature = "ir-serde", serde(skip))]
    holds: Vec<ManagedVariable>,
}

impl PartialEq for ReferenceData {
    fn eq(&self, other: &Self) -> bool {
        self.bindings == other.bindings && self.locals == other.locals && self.addressed == other.addressed
    }
}
impl Eq for ReferenceData {}
impl Hash for ReferenceData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bindings.hash(state);
        self.locals.hash(state);
        self.addressed.hash(state);
    }
}

impl ReferenceData {
    fn target(&self, value: Variable) -> Option<ReferenceTarget> {
        self.bindings.iter().find(|(var, _)| *var == value).map(|(_, target)| target.clone())
    }

    fn local(&self, value: Variable) -> Option<Variable> {
        self.locals.iter().find(|(var, _)| *var == value).map(|(_, storage)| *storage)
    }
}

impl Scope {
    pub fn reference_target(&self, value: Variable) -> Option<ReferenceTarget> {
        self.references.borrow().target(value)
    }

    pub fn reference_pointer(&mut self, mut value: Variable) -> Option<Variable> {
        loop {
            match self.reference_target(value)? {
                ReferenceTarget::Memory(pointer) => return Some(pointer),
                ReferenceTarget::Binding { .. } => {
                    value = *self.reference_read(value).expect("valid nested reference");
                }
            }
        }
    }

    pub fn reference_storage(&self, value: Variable) -> Option<Variable> {
        self.references.borrow().local(value)
    }

    fn reference_proxy(&mut self, ty: Type, target: ReferenceTarget) -> ManagedVariable {
        let value = self.create_local(ty);
        self.references.borrow_mut().bindings.push((*value, target));
        value
    }

    fn reference_slot(&mut self, ty: Type, kind: ReferenceKind, mutable: bool) -> ManagedVariable {
        let payload = if mutable {
            let value = self.create_local_restricted(kind.payload_type());
            self.add_local_mut(*value);
            value
        } else {
            self.create_local(kind.payload_type())
        };
        self.reference_proxy(ty, ReferenceTarget::from_payload(kind, *payload))
    }

    pub fn reference_address(&mut self, pointer: TypedPointer) -> Result<ManagedVariable, String> {
        pointer.validate()?;
        let pointee = pointer.ty.pointee;
        let address = self.create_local(pointer.ty.value_type());
        self.register(Instruction::new(PointerOp::Address(pointer), *address));
        Ok(self.reference_proxy(pointee, ReferenceTarget::Memory(*address)))
    }

    pub fn reference_copy(&mut self, value: Variable, mutable: bool) -> Option<ManagedVariable> {
        let target = self.reference_target(value)?;
        let copy = self.reference_slot(value.ty, target.kind(), mutable);
        let output = self.reference_target(*copy).unwrap().payload();
        self.register(Instruction::new(Operation::Copy(target.payload()), output));
        Some(copy)
    }

    pub fn reference_assign(&mut self, output: Variable, value: Variable) -> Result<(), String> {
        match (self.reference_target(output), self.reference_target(value)) {
            (Some(ReferenceTarget::Memory(out)), Some(ReferenceTarget::Memory(mut input))) => {
                if out.ty != input.ty {
                    let destination = out.ty.pointer_type().expect("reference pointer");
                    let mut source = input.ty.pointer_type().expect("reference pointer");
                    if source.access != PointerAccess::ReadWrite || destination.access != PointerAccess::ReadOnly {
                        return Err("reference assignment requires compatible pointee, space and permissions".into());
                    }
                    source.access = PointerAccess::ReadOnly;
                    if source != destination {
                        return Err("reference assignment requires matching pointee and address space".into());
                    }
                    let narrowed = self.create_local(out.ty);
                    self.register(Instruction::new(PointerOp::ReadOnly(UnaryOperator { input }), *narrowed));
                    input = *narrowed;
                }
                if !matches!(out.kind, VariableKind::LocalMut { .. }) {
                    return Err("reference binding is not mutable".into());
                }
                self.register(Instruction::new(Operation::Copy(input), out));
                Ok(())
            }
            (Some(out @ ReferenceTarget::Binding { .. }), Some(input @ ReferenceTarget::Binding { .. })) => {
                if !out.kind().accepts(&input.kind()) {
                    return Err("nested reference assignment requires compatible target types and permissions".into());
                }
                if !matches!(out.payload().kind, VariableKind::LocalMut { .. }) {
                    return Err("nested reference binding is not mutable".into());
                }
                self.register(Instruction::new(Operation::Copy(input.payload()), out.payload()));
                Ok(())
            }
            _ => Err("reference assignment requires two references".into()),
        }
    }

    pub fn reference_read(&mut self, value: Variable) -> Result<ManagedVariable, String> {
        match self.reference_target(value) {
            Some(ReferenceTarget::Memory(pointer)) => {
                let out = self.create_local(value.ty);
                self.register(Instruction::new(PointerOp::Load { pointer, checked: true }, *out));
                Ok(out)
            }
            Some(ReferenceTarget::Binding { target, .. }) => {
                let out = self.reference_slot(value.ty, *target, true);
                self.register(Instruction::new(PointerOp::ReferenceLoad { reference: value }, *out));
                Ok(out)
            }
            None => Ok(ManagedVariable::Plain(value)),
        }
    }

    pub fn reference_write(&mut self, output: Variable, value: Variable) -> Result<(), String> {
        match self.reference_target(output) {
            Some(ReferenceTarget::Memory(pointer)) => {
                let ty = pointer.ty.pointer_type().expect("reference pointer");
                if ty.access != PointerAccess::ReadWrite {
                    return Err("cannot write through an immutable reference".into());
                }
                self.register(Instruction::no_out(PointerOp::Store { pointer, value, checked: true }));
                Ok(())
            }
            Some(ReferenceTarget::Binding { target, writable: true, .. }) => {
                let source = self.reference_target(value).ok_or_else(|| String::from("nested reference store requires a reference value"))?;
                if !target.accepts(&source.kind()) {
                    return Err("nested reference store requires a compatible reference value".into());
                }
                self.register(Instruction::no_out(PointerOp::ReferenceStore { reference: output, value }));
                Ok(())
            }
            Some(ReferenceTarget::Binding { writable: false, .. }) => {
                Err("cannot write through an immutable nested reference".into())
            }
            None => Err("dereference assignment requires a reference".into()),
        }
    }

    pub fn reference_reborrow(&mut self, value: Variable, writable: bool) -> Result<ManagedVariable, String> {
        let Some(target) = self.reference_target(value) else {
            return Err("reborrow requires a reference".into());
        };
        let target = match target {
            ReferenceTarget::Memory(pointer) => {
                let mut ty = pointer.ty.pointer_type().expect("reference pointer");
                if writable && ty.access != PointerAccess::ReadWrite {
                    return Err("cannot mutably reborrow an immutable reference".into());
                }
                if !writable && !ty.storage.is_atomic() && ty.access == PointerAccess::ReadWrite {
                    ty.access = PointerAccess::ReadOnly;
                    let out = self.create_local(Type::Pointer(ty));
                    self.register(Instruction::new(PointerOp::ReadOnly(UnaryOperator { input: pointer }), *out));
                    ReferenceTarget::Memory(*out)
                } else {
                    ReferenceTarget::Memory(pointer)
                }
            }
            ReferenceTarget::Binding { selector, target, writable: old } => {
                if writable && !old {
                    return Err("cannot mutably reborrow an immutable nested reference".into());
                }
                ReferenceTarget::Binding { selector, target, writable }
            }
        };
        Ok(self.reference_proxy(value.ty, target))
    }

    pub fn reference_borrow(&mut self, value: ManagedVariable, writable: bool) -> Result<ManagedVariable, String> {
        if let Some(target) = self.reference_target(*value) {
            if writable && !matches!(target.payload().kind, VariableKind::LocalMut { .. }) {
                return Err("mutable borrowing requires a mutable reference binding".into());
            }
            let selector = {
                let mut data = self.references.borrow_mut();
                let index = if let Some(index) = data.addressed.iter().position(|(variable, _)| *variable == *value) {
                    data.addressed[index].1 |= writable;
                    index
                } else {
                    data.addressed.push((*value, writable));
                    data.addressed.len() - 1
                };
                Variable::from(u32::try_from(index).expect("reference binding index fits U32"))
            };
            return Ok(self.reference_proxy(value.ty, ReferenceTarget::Binding {
                selector, target: Box::new(target.kind()), writable,
            }));
        }
        if value.is_array() || value.ty.is_semantic() || value.ty.is_pointer() {
            return Ok(value);
        }
        let original = *value;
        let stored = self.references.borrow().local(original);
        let storage = if let Some(storage) = stored {
            storage
        } else {
            let storage = match original.kind {
                VariableKind::Shared { id } => Variable::new(VariableKind::SharedArray {
                    id, length: 1, unroll_factor: 1, alignment: None,
                }, original.ty),
                _ => *self.create_local_array(original.ty, 1),
            };
            if matches!(original.kind, VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } | VariableKind::Shared { .. }) {
                let mut data = self.references.borrow_mut();
                data.locals.push((original, storage));
                data.holds.push(value);
            } else {
                self.register(Instruction::new(Operator::UncheckedIndexAssign(IndexAssignOperator {
                    index: 0u32.into(), value: original, vector_size: 0, unroll_factor: 1,
                }), storage));
            }
            storage
        };
        self.reference_address(TypedPointer {
            base: storage, index: 0u32.into(),
            ty: PointerType {
                pointee: original.ty,
                address_space: if matches!(storage.kind, VariableKind::SharedArray { .. }) {
                    AddressSpace::Shared
                } else { AddressSpace::Local },
                access: if writable || original.ty.is_atomic() { PointerAccess::ReadWrite } else { PointerAccess::ReadOnly },
            },
            checked: true, unroll_factor: 1,
        })
    }

    pub fn lower_reference_values(&mut self) -> Result<(), String> {
        let data = self.references.borrow().clone();
        if data.bindings.is_empty() && data.locals.is_empty() {
            return Ok(());
        }
        ReferenceLowering { data }.scope(self)
    }
}

struct ReferenceLowering {
    data: ReferenceData,
}

impl ReferenceLowering {
    fn read(&self, scope: &mut Scope, value: Variable) -> Variable {
        if let Some(pointer) = self.data.pointer(value) {
            let out = scope.create_local(value.ty);
            scope.register(Instruction::new(PointerOp::Load { pointer, checked: true }, *out));
            *out
        } else if let Some(storage) = self.data.local(value) {
            let out = scope.create_local(value.ty);
            scope.register(Instruction::new(Operator::UncheckedIndex(IndexOperator {
                list: storage, index: 0u32.into(), vector_size: 0, unroll_factor: 1,
            }), *out));
            *out
        } else {
            value
        }
    }

    fn scope(&self, scope: &mut Scope) -> Result<(), String> {
        let instructions = core::mem::take(&mut scope.instructions);
        scope.locals.retain(|var| self.data.local(*var).is_none());
        for parameter in scope.parameters.clone() {
            if let Some(storage) = self.data.local(parameter) {
                scope.register(Instruction::new(Operator::UncheckedIndexAssign(IndexAssignOperator {
                    index: 0u32.into(), value: parameter, vector_size: 0, unroll_factor: 1,
                }), storage));
            }
        }
        for inst in instructions {
            let start = scope.instructions.len();
            let mut operation = inst.operation.clone();
            match &mut operation {
                Operation::Branch(branch) => match branch {
                    Branch::If(b) => {
                        b.cond = self.read(scope, b.cond);
                        self.scope(&mut b.scope)?;
                    }
                    Branch::IfElse(b) => {
                        b.cond = self.read(scope, b.cond);
                        self.scope(&mut b.scope_if)?;
                        self.scope(&mut b.scope_else)?;
                    }
                    Branch::Switch(b) => {
                        b.value = self.read(scope, b.value);
                        self.scope(&mut b.scope_default)?;
                        for (_, child) in &mut b.cases { self.scope(child)?; }
                    }
                    Branch::RangeLoop(b) => {
                        b.start = self.read(scope, b.start);
                        b.end = self.read(scope, b.end);
                        if let Some(step) = &mut b.step { *step = self.read(scope, *step); }
                        self.scope(&mut b.scope)?;
                        if let Some(storage) = self.data.local(b.i) {
                            b.scope.instructions.insert(0, Instruction::new(
                                Operator::UncheckedIndexAssign(IndexAssignOperator {
                                    index: 0u32.into(), value: b.i, vector_size: 0, unroll_factor: 1,
                                }), storage,
                            ));
                        }
                    }
                    Branch::Loop(b) => self.scope(&mut b.scope)?,
                    Branch::ReturnValue(value) => *value = self.read(scope, *value),
                    _ => {}
                },
                Operation::Marker(Marker::Free(value)) => {
                    if let Some(storage) = self.data.local(*value) { *value = storage; }
                }
                Operation::NonSemantic(NonSemantic::Print { args, .. }) => {
                    for value in args { *value = self.read(scope, *value); }
                }
                Operation::CoopMma(CoopMma::Elementwise { function, captures, .. }) => {
                    let mut data = self.data.clone();
                    for (index, value) in captures.iter_mut().enumerate() {
                        if let Some(pointer) = self.data.pointer(*value) {
                            let parameter = &mut function.body.parameters[function.num_arguments + index];
                            let formal = *parameter;
                            parameter.ty = pointer.ty;
                            data.bindings.retain(|(value, _)| *value != formal);
                            data.bindings.push((formal, ReferenceTarget::Memory(*parameter)));
                            *value = pointer;
                        } else {
                            *value = self.read(scope, *value);
                        }
                    }
                    ReferenceLowering { data }.scope(&mut function.body)?;
                }
                _ => {
                    if let Some(args) = operation.args() {
                        let values = args.into_iter().map(|value| self.read(scope, value)).collect::<Vec<_>>();
                        operation = Operation::from_code_and_args(operation.op_code(), &values)
                            .ok_or_else(|| format!("cannot rebuild reference operand in {operation}"))?;
                    }
                }
            }
            let mut out = inst.out;
            if matches!(operation, Operation::Atomic(AtomicOp::Store(_))) {
                out = out.map(|value| self.read(scope, value));
            }
            let storage = out.and_then(|value| self.data.local(value));
            if let Some(original) = inst.out.filter(|_| storage.is_some()) {
                let value = scope.create_local_restricted(original.ty);
                scope.add_local_mut(*value);
                if matches!(operation, Operation::Operator(Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_))) {
                    let previous = self.read(scope, original);
                    scope.register(Instruction::new(Operation::Copy(previous), *value));
                }
                out = Some(*value);
            }
            scope.instructions.push(Instruction { operation, out, ..inst.clone() });
            if let Some(storage) = storage {
                scope.register(Instruction::new(Operator::UncheckedIndexAssign(IndexAssignOperator {
                    index: 0u32.into(), value: out.unwrap(), vector_size: 0, unroll_factor: 1,
                }), storage));
            }
            for emitted in &mut scope.instructions[start..] {
                emitted.source_loc = inst.source_loc.clone();
                emitted.modes = inst.modes;
            }
        }
        scope.references = Default::default();
        Ok(())
    }
}
