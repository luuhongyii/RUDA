use alloc::{format, string::String, vec, vec::Vec};
use core::fmt::{self, Display};

use super::{
    IndexAssignOperator, IndexOperator, Instruction, Operation, OperationReflect, Operator, Type,
    TypeHash, Variable, VariableKind,
};

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    /// Reference values whose concrete storage space is selected by their origin.
    Generic,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PointerAccess {
    ReadOnly,
    ReadWrite,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PointerType {
    pub pointee: Type,
    pub address_space: AddressSpace,
    pub access: PointerAccess,
}

/// Non-recursive, copyable representation of a first-class pointer's type.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PointerValueType {
    pub storage: super::StorageType,
    pub width: usize,
    pub address_space: AddressSpace,
    pub access: PointerAccess,
}

impl PointerValueType {
    pub fn pointee(self) -> Type {
        Type::new(self.storage).with_vector_size(self.width)
    }

    pub fn view(self) -> PointerType {
        PointerType { pointee: self.pointee(), address_space: self.address_space, access: self.access }
    }
}

impl PointerType {
    pub fn value_type(self) -> Type {
        assert!(!self.pointee.is_pointer() && !self.pointee.is_semantic(), "pointer pointee must have storage");
        Type::Pointer(PointerValueType {
            storage: self.pointee.storage_type(), width: self.pointee.vector_size(),
            address_space: self.address_space, access: self.access,
        })
    }
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
pub struct TypedPointer {
    pub base: Variable,
    pub index: Variable,
    pub ty: PointerType,
    pub checked: bool,
    pub unroll_factor: usize,
}

impl TypedPointer {
    pub fn validate(&self) -> Result<(), String> {
        let space = match self.base.kind {
            VariableKind::GlobalInputArray(_) | VariableKind::GlobalOutputArray(_) => {
                AddressSpace::Global
            }
            VariableKind::SharedArray { .. } => AddressSpace::Shared,
            VariableKind::LocalArray { .. } | VariableKind::ConstantArray { .. } => {
                AddressSpace::Local
            }
            _ => {
                return Err(format!(
                    "typed pointer requires array storage, got {}",
                    self.base
                ));
            }
        };
        if space != self.ty.address_space {
            return Err(format!(
                "typed pointer address space mismatch: {:?} != {:?}",
                self.ty.address_space, space
            ));
        }
        if self.ty.pointee.is_pointer() || self.base.ty.is_pointer() || self.ty.pointee.is_semantic()
            || self.base.ty.is_semantic()
            || self.base.ty.storage_type() != self.ty.pointee.storage_type()
        {
            return Err(format!(
                "typed pointer pointee mismatch: {} != {}",
                self.ty.pointee, self.base.ty
            ));
        }
        if self.unroll_factor == 0 {
            return Err("typed pointer unroll factor must be nonzero".into());
        }
        if self.index.ty.vector_size() != 1 || !self.index.ty.is_unsigned_int() {
            return Err("typed pointer element offset must be a scalar unsigned integer".into());
        }
        if self.ty.access == PointerAccess::ReadWrite
            && matches!(self.base.kind, VariableKind::ConstantArray { .. })
        {
            return Err("constant array cannot be used through a writable typed pointer".into());
        }
        Ok(())
    }

    pub fn read_only(self) -> Self {
        Self {
            ty: PointerType {
                access: PointerAccess::ReadOnly,
                ..self.ty
            },
            ..self
        }
    }
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
pub enum TypedMemory {
    Load(TypedPointer),
    Store {
        pointer: TypedPointer,
        value: Variable,
    },
}

#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypedMemoryOpCode {
    Load(PointerType, bool, usize),
    Store(PointerType, bool, usize),
}

impl OperationReflect for TypedMemory {
    type OpCode = TypedMemoryOpCode;
    fn op_code(&self) -> Self::OpCode {
        let ptr = self.pointer();
        match self {
            Self::Load(_) => TypedMemoryOpCode::Load(ptr.ty, ptr.checked, ptr.unroll_factor),
            Self::Store { .. } => TypedMemoryOpCode::Store(ptr.ty, ptr.checked, ptr.unroll_factor),
        }
    }
    fn args(&self) -> Option<Vec<Variable>> {
        let ptr = self.pointer();
        Some(match self {
            Self::Load(_) => vec![ptr.base, ptr.index],
            Self::Store { value, .. } => vec![ptr.base, ptr.index, *value],
        })
    }
    fn from_code_and_args(code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        let (ty, checked, unroll_factor) = match code {
            TypedMemoryOpCode::Load(ty, checked, factor)
            | TypedMemoryOpCode::Store(ty, checked, factor) => (ty, checked, factor),
        };
        let pointer = TypedPointer {
            base: *args.first()?,
            index: *args.get(1)?,
            ty,
            checked,
            unroll_factor,
        };
        match code {
            TypedMemoryOpCode::Load(..) if args.len() == 2 => Some(Self::Load(pointer)),
            TypedMemoryOpCode::Store(..) if args.len() == 3 => Some(Self::Store {
                pointer,
                value: args[2],
            }),
            _ => None,
        }
    }
}

impl TypedMemory {
    pub fn pointer(&self) -> &TypedPointer {
        match self {
            Self::Load(pointer) | Self::Store { pointer, .. } => pointer,
        }
    }

    pub fn validate(&self, out: Option<Variable>) -> Result<(), String> {
        let pointer = self.pointer();
        pointer.validate()?;
        if pointer.ty.pointee.is_atomic() {
            return Err("typed load/store requires a non-atomic pointee".into());
        }
        match self {
            Self::Load(_) => {
                if out.map(|out| out.ty) != Some(pointer.ty.pointee) {
                    return Err("typed load result must match its pointee type".into());
                }
            }
            Self::Store { value, .. } => {
                if pointer.ty.access != PointerAccess::ReadWrite {
                    return Err("cannot store through a read-only typed pointer".into());
                }
                if value.ty != pointer.ty.pointee || out != Some(pointer.base) {
                    return Err("typed store value and destination must match its pointer".into());
                }
            }
        }
        Ok(())
    }

    pub fn lower(&self, out: Option<Variable>) -> Result<Instruction, String> {
        self.validate(out)?;
        let pointer = self.pointer();
        let vector_size = if pointer.ty.pointee.vector_size() == pointer.base.ty.vector_size() {
            0
        } else {
            pointer.ty.pointee.vector_size()
        };
        let op = match self {
            Self::Load(_) => {
                let op = IndexOperator {
                    list: pointer.base,
                    index: pointer.index,
                    vector_size,
                    unroll_factor: pointer.unroll_factor,
                };
                if pointer.checked {
                    Operator::Index(op)
                } else {
                    Operator::UncheckedIndex(op)
                }
            }
            Self::Store { value, .. } => {
                let op = IndexAssignOperator {
                    index: pointer.index,
                    value: *value,
                    vector_size,
                    unroll_factor: pointer.unroll_factor,
                };
                if pointer.checked {
                    Operator::IndexAssign(op)
                } else {
                    Operator::UncheckedIndexAssign(op)
                }
            }
        };
        Ok(Instruction::new(op, out.unwrap()))
    }

    pub fn retag(&self, mut instruction: Instruction) -> Instruction {
        let original = self.pointer();
        let same_storage = |base: Variable| {
            core::mem::discriminant(&base.kind) == core::mem::discriminant(&original.base.kind)
                && base.index() == original.base.index()
        };
        let replacement = match &instruction.operation {
            Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op))
                if matches!(self, Self::Load(_)) && same_storage(op.list) =>
            {
                let pointer = TypedPointer {
                    base: op.list,
                    index: op.index,
                    ty: PointerType {
                        pointee: instruction.out().ty,
                        ..original.ty
                    },
                    checked: matches!(
                        instruction.operation,
                        Operation::Operator(Operator::Index(_))
                    ),
                    unroll_factor: op.unroll_factor,
                };
                Some(Self::Load(pointer))
            }
            Operation::Operator(Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op))
                if matches!(self, Self::Store { .. })
                    && instruction.out.is_some_and(same_storage) =>
            {
                let pointer = TypedPointer {
                    base: instruction.out(),
                    index: op.index,
                    ty: PointerType {
                        pointee: op.value.ty,
                        ..original.ty
                    },
                    checked: matches!(
                        instruction.operation,
                        Operation::Operator(Operator::IndexAssign(_))
                    ),
                    unroll_factor: op.unroll_factor,
                };
                Some(Self::Store {
                    pointer,
                    value: op.value,
                })
            }
            _ => None,
        };
        if let Some(memory) = replacement {
            instruction.operation = Operation::TypedMemory(memory);
        }
        if let Operation::Branch(branch) = &mut instruction.operation {
            let mut retag_scope = |scope: &mut super::Scope| {
                for instruction in &mut scope.instructions {
                    *instruction = self.retag(instruction.clone());
                }
            };
            match branch {
                super::Branch::If(branch) => retag_scope(&mut branch.scope),
                super::Branch::IfElse(branch) => {
                    retag_scope(&mut branch.scope_if);
                    retag_scope(&mut branch.scope_else);
                }
                super::Branch::Switch(branch) => {
                    retag_scope(&mut branch.scope_default);
                    for (_, scope) in &mut branch.cases {
                        retag_scope(scope);
                    }
                }
                super::Branch::RangeLoop(branch) => retag_scope(&mut branch.scope),
                super::Branch::Loop(branch) => retag_scope(&mut branch.scope),
                _ => {}
            }
        }
        instruction
    }
}

impl Display for TypedMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(ptr) => write!(
                f,
                "typed_load({}[{}], {:?}, {:?})",
                ptr.base, ptr.index, ptr.ty.address_space, ptr.ty.access
            ),
            Self::Store {
                pointer: ptr,
                value,
            } => write!(
                f,
                "typed_store({}[{}], {}, {:?}, {:?})",
                ptr.base, ptr.index, value, ptr.ty.address_space, ptr.ty.access
            ),
        }
    }
}

impl Instruction {
    pub fn retag_typed_memory(&self, replacement: Self) -> Self {
        match &self.operation {
            Operation::TypedMemory(memory) => memory.retag(replacement),
            _ => replacement,
        }
    }
    pub fn lower_typed_memory(mut self) -> Result<Self, String> {
        if let Operation::TypedMemory(op) = &self.operation {
            self.operation = op.lower(self.out)?.operation;
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ConstantValue, ElemType, FloatKind};

    fn pointer() -> TypedPointer {
        let pointee = Type::scalar(ElemType::Float(FloatKind::F32));
        TypedPointer {
            base: Variable::new(
                VariableKind::SharedArray {
                    id: 0,
                    length: 4,
                    unroll_factor: 1,
                    alignment: None,
                },
                pointee,
            ),
            index: Variable::constant(
                ConstantValue::UInt(0),
                Type::scalar(ElemType::UInt(super::super::UIntKind::U32)),
            ),
            ty: PointerType {
                pointee,
                address_space: AddressSpace::Shared,
                access: PointerAccess::ReadWrite,
            },
            checked: true,
            unroll_factor: 1,
        }
    }

    #[test]
    fn preserves_type_and_permissions_when_lowering() {
        let ptr = pointer();
        let out = Variable::new(VariableKind::LocalMut { id: 1 }, ptr.ty.pointee);
        assert!(ptr.validate().is_ok());
        assert!(TypedMemory::Load(ptr.read_only()).lower(Some(out)).is_ok());
        assert!(
            TypedMemory::Store {
                pointer: ptr,
                value: out
            }
            .lower(Some(ptr.base))
            .is_ok()
        );
        assert!(
            TypedMemory::Store {
                pointer: ptr.read_only(),
                value: out
            }
            .lower(Some(ptr.base))
            .is_err()
        );
    }

    #[test]
    fn rejects_wrong_space_pointee_and_index() {
        let ptr = pointer();
        let mut invalid = ptr;
        invalid.ty.address_space = AddressSpace::Global;
        assert!(invalid.validate().is_err());
        invalid = ptr;
        invalid.ty.pointee = Type::scalar(ElemType::Bool);
        assert!(invalid.validate().is_err());
        invalid = ptr;
        invalid.index.ty = Type::scalar(ElemType::Float(FloatKind::F32));
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn reflection_preserves_pointer_metadata() {
        let mut pointer = pointer();
        pointer.checked = false;
        pointer.unroll_factor = 2;
        let operation = TypedMemory::Load(pointer.read_only());
        assert_eq!(
            TypedMemory::from_code_and_args(operation.op_code(), &operation.args().unwrap()),
            Some(operation)
        );
    }

    #[test]
    fn constant_storage_is_read_only() {
        let mut ptr = pointer();
        ptr.base.kind = VariableKind::ConstantArray {
            id: 0,
            length: 4,
            unroll_factor: 1,
        };
        ptr.ty.address_space = AddressSpace::Local;
        assert!(ptr.validate().is_err());
        assert!(ptr.read_only().validate().is_ok());
    }
}
