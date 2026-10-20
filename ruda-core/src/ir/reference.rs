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

    pub fn reference_component(&mut self, value: Variable, index: Variable, writable: bool)
        -> Result<ManagedVariable, String>
    {
        let pointer = self.reference_pointer(value).ok_or("component reference requires vector storage")?;
        let mut ty = pointer.ty.pointer_type().ok_or("component reference requires a pointer")?;
        if writable && ty.access != PointerAccess::ReadWrite {
            return Err("cannot mutably borrow an immutable vector component".into());
        }
        ty.width = 1;
        if !writable && !ty.storage.is_atomic() { ty.access = PointerAccess::ReadOnly; }
        let out = self.create_local(Type::Pointer(ty));
        self.register(Instruction::new(PointerOp::Component(BinaryOperator { lhs: pointer, rhs: index }), *out));
        Ok(self.reference_proxy(ty.pointee(), ReferenceTarget::Memory(*out)))
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
        let mut ty = address.ty.pointer_type().expect("reference pointer");
        ty.address_space = AddressSpace::Generic;
        let erased = self.create_local(Type::Pointer(ty));
        self.register(Instruction::new(PointerOp::EraseSpace(UnaryOperator { input: *address }), *erased));
        Ok(self.reference_proxy(pointee, ReferenceTarget::Memory(*erased)))
    }

    pub fn reference_absent(&mut self, pointee: Type, writable: bool) -> ManagedVariable {
        let ty = PointerType {
            pointee, address_space: AddressSpace::Generic,
            access: if writable || pointee.is_atomic() { PointerAccess::ReadWrite } else { PointerAccess::ReadOnly },
        }.value_type();
        let pointer = self.create_local(ty);
        self.register(Instruction::new(PointerOp::Absent, *pointer));
        self.reference_proxy(pointee, ReferenceTarget::Memory(*pointer))
    }

    pub fn reference_absent_binding(&mut self, template: Variable, writable: bool) -> Result<ManagedVariable, String> {
        let target = self.reference_target(template).ok_or("nested absent reference requires a reference type")?;
        Ok(self.reference_proxy(template.ty, ReferenceTarget::Binding {
            selector: u32::MAX.into(), target: Box::new(target.kind()), writable,
        }))
    }

    pub fn reference_copy(&mut self, value: Variable, mutable: bool) -> Option<ManagedVariable> {
        let target = self.reference_target(value)?;
        let copy = self.reference_slot(value.ty, target.kind(), mutable);
        let output = self.reference_target(*copy).unwrap().payload();
        self.register(Instruction::new(Operation::Copy(target.payload()), output));
        Some(copy)
    }

    pub fn reference_empty_mut(&mut self, value: Variable) -> Option<ManagedVariable> {
        let kind = self.reference_target(value)?.kind();
        Some(self.reference_slot(value.ty, kind, true))
    }

    pub fn reference_join(&mut self, a: Variable, b: Variable) -> Option<ManagedVariable> {
        let ty = a.ty;
        let a = self.reference_target(a)?;
        let b = self.reference_target(b)?;
        let kind = match (a.kind(), b.kind()) {
            (ReferenceKind::Memory(mut a), ReferenceKind::Memory(b)) => {
                if b.access == PointerAccess::ReadOnly { a.access = PointerAccess::ReadOnly; }
                let kind = ReferenceKind::Memory(a);
                assert!(kind.accepts(&ReferenceKind::Memory(b)), "incompatible branch references");
                kind
            }
            (ReferenceKind::Binding { target, writable }, ReferenceKind::Binding { target: other, writable: rw }) => {
                assert_eq!(target, other, "incompatible nested branch references");
                ReferenceKind::Binding { target, writable: writable && rw }
            }
            _ => panic!("branch references have different depths"),
        };
        Some(self.reference_slot(ty, kind, true))
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
        if let Some(mut target) = self.reference_target(*value) {
            if writable && !matches!(target.payload().kind, VariableKind::LocalMut { .. }) {
                let payload = self.create_local_restricted(target.kind().payload_type());
                self.add_local_mut(*payload);
                self.register(Instruction::new(Operation::Copy(target.payload()), *payload));
                target = ReferenceTarget::from_payload(target.kind(), *payload);
                let mut data = self.references.borrow_mut();
                let entry = data.bindings.iter_mut().find(|(var, _)| *var == *value).unwrap();
                entry.1 = target.clone();
                data.holds.push(value.clone());
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
                Variable::from(u32::try_from(index).ok().filter(|index| *index != u32::MAX)
                    .expect("reference binding index fits below the absent selector"))
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
        let mut lowering = ReferenceLowering { data, selectors: Vec::new(), absent_selectors: Vec::new() };
        while lowering.collect_selectors(self) {}
        lowering.scope(self)
    }
}

struct ReferenceLowering {
    data: ReferenceData,
    selectors: Vec<(Variable, Vec<usize>)>,
    absent_selectors: Vec<Variable>,
}

impl ReferenceLowering {
    fn is_absent(&self, selector: Variable) -> bool {
        selector.as_const().and_then(|value| value.try_as_usize()) == Some(u32::MAX as usize)
            || self.absent_selectors.contains(&selector)
    }

    fn extend_absent(&mut self, selector: Variable) -> bool {
        if selector.as_const().is_some() || self.absent_selectors.contains(&selector) { return false; }
        self.absent_selectors.push(selector);
        true
    }

    fn targets(&self, selector: Variable) -> Vec<usize> {
        if let Some(index) = selector.as_const().and_then(|value| value.try_as_usize()) {
            return if index < self.data.addressed.len() { alloc::vec![index] } else { Vec::new() };
        }
        self.selectors.iter().find(|(value, _)| *value == selector)
            .map(|(_, targets)| targets.clone()).unwrap_or_default()
    }

    fn extend_targets(&mut self, selector: Variable, targets: Vec<usize>) -> bool {
        if targets.is_empty() || selector.as_const().is_some() { return false; }
        let index = self.selectors.iter().position(|(value, _)| *value == selector)
            .unwrap_or_else(|| {
                self.selectors.push((selector, Vec::new()));
                self.selectors.len() - 1
            });
        let known = &mut self.selectors[index].1;
        let before = known.len();
        for target in targets {
            if !known.contains(&target) { known.push(target); }
        }
        known.len() != before
    }

    fn collect_selectors(&mut self, scope: &Scope) -> bool {
        let mut changed = false;
        for instruction in &scope.instructions {
            match &instruction.operation {
                Operation::Copy(input) => {
                    if let Some(out) = instruction.out {
                        changed |= self.extend_targets(out, self.targets(*input));
                        if self.is_absent(*input) { changed |= self.extend_absent(out); }
                    }
                }
                Operation::Pointer(PointerOp::ReferenceLoad { reference }) => {
                    if let (Some(ReferenceTarget::Binding { selector, .. }), Some(output)) =
                        (self.data.target(*reference), instruction.out.and_then(|out| self.data.target(out)))
                    {
                        if matches!(output, ReferenceTarget::Binding { .. }) {
                            if self.is_absent(selector) { changed |= self.extend_absent(output.payload()); }
                            for index in self.targets(selector) {
                                if let Some(ReferenceTarget::Binding { selector, .. }) =
                                    self.data.target(self.data.addressed[index].0)
                                {
                                    changed |= self.extend_targets(output.payload(), self.targets(selector));
                                    if self.is_absent(selector) { changed |= self.extend_absent(output.payload()); }
                                }
                            }
                        }
                    }
                }
                Operation::Pointer(PointerOp::ReferenceStore { reference, value }) => {
                    if let (Some(ReferenceTarget::Binding { selector, .. }),
                            Some(ReferenceTarget::Binding { selector: source, .. })) =
                        (self.data.target(*reference), self.data.target(*value))
                    {
                        for index in self.targets(selector) {
                            if let Some(ReferenceTarget::Binding { selector, .. }) =
                                self.data.target(self.data.addressed[index].0)
                            {
                                changed |= self.extend_targets(selector, self.targets(source));
                                if self.is_absent(source) { changed |= self.extend_absent(selector); }
                            }
                        }
                    }
                }
                Operation::Branch(branch) => match branch {
                    Branch::If(b) => changed |= self.collect_selectors(&b.scope),
                    Branch::IfElse(b) => {
                        changed |= self.collect_selectors(&b.scope_if);
                        changed |= self.collect_selectors(&b.scope_else);
                    }
                    Branch::Switch(b) => {
                        changed |= self.collect_selectors(&b.scope_default);
                        for (_, child) in &b.cases { changed |= self.collect_selectors(child); }
                    }
                    Branch::RangeLoop(b) => changed |= self.collect_selectors(&b.scope),
                    Branch::Loop(b) => changed |= self.collect_selectors(&b.scope),
                    _ => {}
                },
                Operation::CoopMma(CoopMma::Elementwise { function, .. }) => {
                    changed |= self.collect_selectors(&function.body);
                }
                _ => {}
            }
        }
        changed
    }

    fn copy_payload(scope: &mut Scope, source: Variable, output: Variable) {
        let narrow = match (source.ty.pointer_type(), output.ty.pointer_type()) {
            (Some(a), Some(b)) => a.access == PointerAccess::ReadWrite && b.access == PointerAccess::ReadOnly,
            _ => false,
        };
        if narrow {
            scope.register(Instruction::new(PointerOp::ReadOnly(UnaryOperator { input: source }), output));
        } else {
            scope.register(Instruction::new(Operation::Copy(source), output));
        }
    }

    fn access_binding(&mut self, scope: &mut Scope, selector: Variable, kind: &ReferenceKind,
        output: Option<Variable>, source: Option<ReferenceTarget>) -> Result<(), String>
    {
        let targets = self.targets(selector);
        if targets.is_empty() {
            if !self.is_absent(selector) { return Err("nested reference has no reachable binding".into()); }
            if let Some(output) = output {
                if output.ty.is_pointer() {
                    scope.register(Instruction::new(PointerOp::Absent, output));
                } else {
                    scope.register(Instruction::new(Operation::Copy(u32::MAX.into()), output));
                }
            }
            scope.register(Instruction::no_out(Branch::Unreachable));
            return Ok(());
        }
        let mut cases = Vec::new();
        for index in targets {
            let (binding, writable) = self.data.addressed[index];
            let destination = self.data.target(binding).ok_or("missing nested reference target")?;
            if !kind.accepts(&destination.kind()) {
                return Err("nested reference target has incompatible storage or reference type".into());
            }
            let mut child = scope.child();
            child.references = Default::default();
            match (&source, output) {
                (Some(source), None) => {
                    if !writable || !matches!(destination.payload().kind, VariableKind::LocalMut { .. }) {
                        return Err("nested reference store requires a mutable target binding".into());
                    }
                    if !destination.kind().accepts(&source.kind()) {
                        return Err("nested reference assignment changes reference type".into());
                    }
                    Self::copy_payload(&mut child, source.payload(), destination.payload());
                }
                (None, Some(output)) => {
                    Self::copy_payload(&mut child, destination.payload(), output);
                    if matches!(destination, ReferenceTarget::Binding { .. }) {
                        self.extend_targets(output, self.targets(destination.payload()));
                        if self.is_absent(destination.payload()) { self.extend_absent(output); }
                    }
                }
                _ => return Err("invalid nested reference access".into()),
            }
            if selector.as_const().is_some() {
                scope.instructions.extend(child.instructions);
                return Ok(());
            }
            cases.push((Variable::from(index as u32), child));
        }
        let mut scope_default = scope.child();
        scope_default.references = Default::default();
        scope_default.register(Instruction::no_out(Branch::Unreachable));
        scope.register(Instruction::no_out(Branch::Switch(Box::new(Switch {
            value: selector, scope_default, cases,
        }))));
        Ok(())
    }

    fn pointer(&mut self, scope: &mut Scope, value: Variable) -> Result<Option<Variable>, String> {
        let Some(mut target) = self.data.target(value) else { return Ok(None); };
        loop {
            target = match target {
                ReferenceTarget::Memory(pointer) => return Ok(Some(pointer)),
                ReferenceTarget::Binding { selector, target, .. } => {
                    let result = scope.create_local_restricted(target.payload_type());
                    scope.add_local_mut(*result);
                    self.access_binding(scope, selector, &target, Some(*result), None)?;
                    ReferenceTarget::from_payload(*target, *result)
                }
            };
        }
    }

    fn read(&mut self, scope: &mut Scope, value: Variable) -> Result<Variable, String> {
        Ok(if let Some(pointer) = self.pointer(scope, value)? {
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
        })
    }

    fn scope(&mut self, scope: &mut Scope) -> Result<(), String> {
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
            match &operation {
                Operation::Atomic(atomic) => {
                    let mut args = atomic.args().ok_or("atomic operation is missing its operands")?;
                    let storage = if atomic.op_code() == AtomicOpCode::Store {
                        inst.out.ok_or("atomic store is missing its destination")?
                    } else {
                        if args.is_empty() { return Err("atomic operation is missing its address".into()); }
                        args.remove(0)
                    };
                    let pointer = self.pointer(scope, storage)?;
                    if let Some(pointer) = pointer {
                        let values = args.into_iter().map(|value| self.read(scope, value)).collect::<Result<Vec<_>, _>>()?;
                        scope.instructions.push(Instruction {
                            operation: PointerOp::Atomic { pointer, operation: atomic.op_code(), values }.into(),
                            out: if atomic.op_code() == AtomicOpCode::Store { None } else { inst.out },
                            ..inst.clone()
                        });
                        for emitted in &mut scope.instructions[start..] {
                            emitted.source_loc = inst.source_loc.clone();
                            emitted.modes = inst.modes;
                        }
                        continue;
                    }
                }
                Operation::Pointer(PointerOp::ReferenceLoad { reference }) => {
                    let Some(ReferenceTarget::Binding { selector, target, .. }) = self.data.target(*reference)
                        else { return Err("reference load requires a nested reference".into()); };
                    let output = inst.out.and_then(|out| self.data.target(out))
                        .ok_or("reference load requires a reference output")?;
                    self.access_binding(scope, selector, &target, Some(output.payload()), None)?;
                    for emitted in &mut scope.instructions[start..] {
                        emitted.source_loc = inst.source_loc.clone();
                        emitted.modes = inst.modes;
                    }
                    continue;
                }
                Operation::Pointer(PointerOp::ReferenceStore { reference, value }) => {
                    let Some(ReferenceTarget::Binding { selector, target, writable: true }) = self.data.target(*reference)
                        else { return Err("reference store requires a writable nested reference".into()); };
                    let source = self.data.target(*value).ok_or("reference store requires a reference source")?;
                    self.access_binding(scope, selector, &target, None, Some(source))?;
                    for emitted in &mut scope.instructions[start..] {
                        emitted.source_loc = inst.source_loc.clone();
                        emitted.modes = inst.modes;
                    }
                    continue;
                }
                _ => {}
            }
            match &mut operation {
                Operation::Branch(branch) => match branch {
                    Branch::If(b) => {
                        b.cond = self.read(scope, b.cond)?;
                        self.scope(&mut b.scope)?;
                    }
                    Branch::IfElse(b) => {
                        b.cond = self.read(scope, b.cond)?;
                        self.scope(&mut b.scope_if)?;
                        self.scope(&mut b.scope_else)?;
                    }
                    Branch::Switch(b) => {
                        b.value = self.read(scope, b.value)?;
                        self.scope(&mut b.scope_default)?;
                        for (_, child) in &mut b.cases { self.scope(child)?; }
                    }
                    Branch::RangeLoop(b) => {
                        b.start = self.read(scope, b.start)?;
                        b.end = self.read(scope, b.end)?;
                        if let Some(step) = &mut b.step { *step = self.read(scope, *step)?; }
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
                    Branch::ReturnValue(value) => *value = self.read(scope, *value)?,
                    _ => {}
                },
                Operation::Marker(Marker::Free(value)) => {
                    if let Some(storage) = self.data.local(*value) { *value = storage; }
                }
                Operation::NonSemantic(NonSemantic::Print { args, .. }) => {
                    for value in args { *value = self.read(scope, *value)?; }
                }
                Operation::CoopMma(CoopMma::Elementwise { function, captures, .. }) => {
                    let mut data = self.data.clone();
                    let mut selectors = self.selectors.clone();
                    let mut absent_selectors = self.absent_selectors.clone();
                    let mut pending = Vec::new();
                    for (index, value) in captures.iter_mut().enumerate() {
                        if let Some(target) = self.data.target(*value) {
                            let parameter = &mut function.body.parameters[function.num_arguments + index];
                            let formal = *parameter;
                            let payload = target.payload();
                            parameter.ty = payload.ty;
                            data.bindings.retain(|(value, _)| *value != formal);
                            data.bindings.push((formal, ReferenceTarget::from_payload(target.kind(), *parameter)));
                            if matches!(target, ReferenceTarget::Binding { .. }) {
                                let targets = self.targets(payload);
                                pending.extend(targets.iter().copied());
                                selectors.push((*parameter, targets));
                                if self.is_absent(payload) { absent_selectors.push(*parameter); }
                            }
                            *value = payload;
                        } else if let Some(storage) = data.local(*value) {
                            function.body.parameters[function.num_arguments + index] = storage;
                            *value = storage;
                        } else {
                            *value = self.read(scope, *value)?;
                        }
                    }
                    let first_argument = function.body.parameters[..function.num_arguments].iter()
                        .filter_map(|var| match var.kind {
                            VariableKind::LocalConst { id } | VariableKind::LocalMut { id } => Some(id),
                            _ => None,
                        }).min().unwrap_or(u32::MAX);
                    let mut visited = Vec::new();
                    while let Some(index) = pending.pop() {
                        if visited.contains(&index) { continue; }
                        visited.push(index);
                        let binding = data.addressed[index].0;
                        let target = data.target(binding).ok_or("missing captured reference binding")?;
                        let payload = target.payload();
                        let external = match binding.kind {
                            VariableKind::LocalConst { id } | VariableKind::LocalMut { id } => id < first_argument,
                            _ => true,
                        };
                        if external && payload.as_const().is_none() && !function.body.parameters.contains(&payload) {
                            function.body.parameters.push(payload);
                            captures.push(payload);
                        }
                        if matches!(target, ReferenceTarget::Binding { .. }) {
                            pending.extend(self.targets(payload));
                        }
                    }
                    let mut lowering = ReferenceLowering { data, selectors, absent_selectors };
                    while lowering.collect_selectors(&function.body) {}
                    lowering.scope(&mut function.body)?;
                }
                _ => {
                    if let Some(args) = operation.args() {
                        let values = args.into_iter().map(|value| self.read(scope, value)).collect::<Result<Vec<_>, _>>()?;
                        operation = Operation::from_code_and_args(operation.op_code(), &values)
                            .ok_or_else(|| format!("cannot rebuild reference operand in {operation}"))?;
                    }
                }
            }
            let mut out = inst.out;
            if matches!(operation, Operation::Atomic(AtomicOp::Store(_))) {
                out = out.map(|value| self.read(scope, value)).transpose()?;
            }
            let storage = out.and_then(|value| self.data.local(value));
            if let Some(original) = inst.out.filter(|_| storage.is_some()) {
                let value = scope.create_local_restricted(original.ty);
                scope.add_local_mut(*value);
                if matches!(operation, Operation::Operator(Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_))) {
                    let previous = self.read(scope, original)?;
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
