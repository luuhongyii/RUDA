use crate::dsl::prelude::*;
use alloc::boxed::Box;
use core::marker::PhantomData;
use ruda_core::ir::{
    AddressSpace, Branch, ElemType, IfElse, Instruction, ManagedVariable, PointerAccess,
    PointerType, TypedMemory, TypedPointer, Variable,
};
use ruda_kernel_macros::{intrinsic, ruda};

#[cfg(test)]
mod tests;

mod value;
pub use value::*;

mod sealed {
    pub trait Sealed {}
}

pub trait PointerSpace: sealed::Sealed + Clone + Copy + Send + Sync + 'static {
    const SPACE: AddressSpace;
}
pub trait PointerPermission: sealed::Sealed + SliceVisibility {
    const ACCESS: PointerAccess;
}

#[derive(Clone, Copy)]
pub struct GlobalSpace;
#[derive(Clone, Copy)]
pub struct Workgroup;
#[derive(Clone, Copy)]
pub struct Local;

impl sealed::Sealed for GlobalSpace {}
impl sealed::Sealed for Workgroup {}
impl sealed::Sealed for Local {}
impl sealed::Sealed for ReadOnly {}
impl sealed::Sealed for ReadWrite {}
impl PointerSpace for GlobalSpace {
    const SPACE: AddressSpace = AddressSpace::Global;
}
impl PointerSpace for Workgroup {
    const SPACE: AddressSpace = AddressSpace::Shared;
}
impl PointerSpace for Local {
    const SPACE: AddressSpace = AddressSpace::Local;
}
impl PointerPermission for ReadOnly {
    const ACCESS: PointerAccess = PointerAccess::ReadOnly;
}
impl PointerPermission for ReadWrite {
    const ACCESS: PointerAccess = PointerAccess::ReadWrite;
}

#[derive(Clone)]
enum PointerNode {
    Element(TypedPointer),
    Select {
        condition: Variable,
        then: Box<PointerNode>,
        otherwise: Box<PointerNode>,
    },
}

/// A logical element pointer whose storage space and permissions are part of its Rust type.
///
/// Read-only pointers cannot store:
/// ```compile_fail
/// use ruda_kernel::dsl::prelude::{GlobalSpace, Ptr};
/// fn write(pointer: Ptr<f32, GlobalSpace>) { pointer.store(1.0); }
/// ```
/// Address spaces cannot be interchanged:
/// ```compile_fail
/// use ruda_kernel::dsl::prelude::{GlobalSpace, Ptr, Workgroup};
/// fn global(_: Ptr<f32, GlobalSpace>) {}
/// fn shared(pointer: Ptr<f32, Workgroup>) { global(pointer); }
/// ```
/// Read-only permissions cannot be upgraded:
/// ```compile_fail
/// use ruda_kernel::dsl::prelude::{GlobalSpace, Ptr, ReadWrite};
/// fn upgrade(pointer: Ptr<f32, GlobalSpace>) -> Ptr<f32, GlobalSpace, ReadWrite> { pointer }
/// ```
#[derive(Clone, Copy)]
pub struct Ptr<E: RudaPrimitive, S: PointerSpace, A: PointerPermission = ReadOnly> {
    marker: PhantomData<(E, S, A)>,
}

pub struct PtrExpand<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> {
    node: PointerNode,
    marker: PhantomData<(E, S, A)>,
}

impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> Clone for PtrExpand<E, S, A> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            marker: PhantomData,
        }
    }
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> IntoMut for PtrExpand<E, S, A> {
    fn into_mut(self, _: &mut Scope) -> Self {
        self
    }
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaDebug for PtrExpand<E, S, A> {}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for Ptr<E, S, A> {
    type ExpandType = PtrExpand<E, S, A>;
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for &Ptr<E, S, A> {
    type ExpandType = PtrExpand<E, S, A>;
}

impl PointerNode {
    fn value(&self, scope: &mut Scope) -> ManagedVariable {
        use ruda_core::ir::{PointerOp, Select};
        let (operation, ty) = match self {
            Self::Element(pointer) => (PointerOp::Address(*pointer), pointer.ty.value_type()),
            Self::Select { condition, then, otherwise } => {
                let then = then.value(scope);
                let otherwise = otherwise.value(scope);
                (PointerOp::Select(Select { cond: *condition, then: *then, or_else: *otherwise }), then.ty)
            }
        };
        let out = scope.create_local(ty);
        scope.register(Instruction::new(operation, *out));
        out
    }
    fn pointee(&self) -> Type {
        match self {
            Self::Element(pointer) => pointer.ty.pointee,
            Self::Select { then, .. } => then.pointee(),
        }
    }

    fn offset(&self, scope: &mut Scope, offset: NativeExpand<usize>) -> Self {
        match self {
            Self::Element(pointer) => {
                let index = add::expand(
                    scope,
                    NativeExpand::<usize>::new(ManagedVariable::Plain(pointer.index)),
                    offset,
                );
                Self::Element(TypedPointer {
                    index: index.expand.consume(),
                    ..*pointer
                })
            }
            Self::Select {
                condition,
                then,
                otherwise,
            } => Self::Select {
                condition: *condition,
                then: Box::new(then.offset(scope, offset.clone())),
                otherwise: Box::new(otherwise.offset(scope, offset)),
            },
        }
    }

    fn read_only(&self) -> Self {
        match self {
            Self::Element(pointer) => Self::Element(pointer.read_only()),
            Self::Select {
                condition,
                then,
                otherwise,
            } => Self::Select {
                condition: *condition,
                then: Box::new(then.read_only()),
                otherwise: Box::new(otherwise.read_only()),
            },
        }
    }

    fn emit(&self, scope: &mut Scope, value: Variable, store: bool) {
        match self {
            Self::Element(pointer) => {
                let (op, out) = if store {
                    (
                        TypedMemory::Store {
                            pointer: *pointer,
                            value,
                        },
                        pointer.base,
                    )
                } else {
                    (TypedMemory::Load(*pointer), value)
                };
                op.validate(Some(out)).expect("valid typed pointer access");
                scope.register(Instruction::new(op, out));
            }
            Self::Select {
                condition,
                then,
                otherwise,
            } => {
                let mut scope_if = scope.child();
                let mut scope_else = scope.child();
                then.emit(&mut scope_if, value, store);
                otherwise.emit(&mut scope_else, value, store);
                scope.register(Instruction::no_out(Branch::IfElse(Box::new(IfElse {
                    cond: *condition,
                    scope_if,
                    scope_else,
                }))));
            }
        }
    }
}

#[ruda]
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> Ptr<E, S, A> {
    /// Materialize a pointer value that can be reassigned and carried across branches and loops.
    pub fn value(&self) -> ValuePtr<E, S, A> {
        intrinsic!(|scope| { NativeExpand::new(self.node.value(scope)) })
    }
    pub fn offset(&self, offset: usize) -> Self {
        intrinsic!(|scope| {
            PtrExpand {
                node: self.node.offset(scope, offset),
                marker: PhantomData,
            }
        })
    }

    pub fn read_only(&self) -> Ptr<E, S, ReadOnly> {
        intrinsic!(|_scope| {
            PtrExpand {
                node: self.node.read_only(),
                marker: PhantomData,
            }
        })
    }

    pub fn select(&self, condition: bool, otherwise: Self) -> Self {
        intrinsic!(|_scope| {
            assert_eq!(
                self.node.pointee(),
                otherwise.node.pointee(),
                "typed pointer selection requires equal pointee types"
            );
            assert_eq!(
                condition.expand.ty,
                Type::scalar(ElemType::Bool),
                "typed pointer selection requires a scalar condition"
            );
            PtrExpand {
                node: PointerNode::Select {
                    condition: condition.expand.consume(),
                    then: Box::new(self.node),
                    otherwise: Box::new(otherwise.node),
                },
                marker: PhantomData,
            }
        })
    }

    pub fn load(&self) -> E {
        intrinsic!(|scope| {
            let out = if matches!(&self.node, PointerNode::Select { .. }) {
                scope.create_local_mut(self.node.pointee())
            } else {
                scope.create_local(self.node.pointee())
            };
            self.node.emit(scope, *out, false);
            NativeExpand::<E>::new(out)
        })
    }
}

#[ruda]
impl<E: RudaPrimitive, S: PointerSpace> Ptr<E, S, ReadWrite> {
    pub fn store(&self, value: E) {
        intrinsic!(|scope| {
            self.node.emit(scope, value.expand.consume(), true);
        })
    }
}

pub struct TypedSlice<E: RudaPrimitive, S: PointerSpace, A: PointerPermission = ReadOnly> {
    marker: PhantomData<(E, S, A)>,
}
pub struct TypedSliceExpand<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> {
    slice: SliceExpand<E, A>,
    marker: PhantomData<S>,
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> Clone for TypedSliceExpand<E, S, A> {
    fn clone(&self) -> Self {
        Self {
            slice: self.slice.clone(),
            marker: PhantomData,
        }
    }
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> IntoMut
    for TypedSliceExpand<E, S, A>
{
    fn into_mut(self, _: &mut Scope) -> Self {
        self
    }
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaDebug
    for TypedSliceExpand<E, S, A>
{
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for TypedSlice<E, S, A> {
    type ExpandType = TypedSliceExpand<E, S, A>;
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for &TypedSlice<E, S, A> {
    type ExpandType = TypedSliceExpand<E, S, A>;
}

#[ruda]
impl<E: RudaPrimitive, A: PointerPermission> Slice<E, A> {
    pub fn typed<S: PointerSpace>(&self) -> TypedSlice<E, S, A> {
        intrinsic!(|_scope| {
            let (base, index) = self.__to_raw_parts();
            let pointee = base
                .ty
                .with_vector_size(self.vector_size.unwrap_or(base.ty.vector_size()));
            TypedPointer {
                base,
                index,
                ty: PointerType {
                    pointee,
                    address_space: S::SPACE,
                    access: A::ACCESS,
                },
                checked: true,
                unroll_factor: 1,
            }
            .validate()
            .expect("valid typed slice conversion");
            TypedSliceExpand {
                slice: self,
                marker: PhantomData,
            }
        })
    }
}

#[ruda]
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> TypedSlice<E, S, A> {
    pub fn ptr(&self, index: usize) -> Ptr<E, S, A> {
        intrinsic!(|scope| {
            let (base, start) = self.slice.__to_raw_parts();
            let index = add::expand(
                scope,
                NativeExpand::<usize>::new(ManagedVariable::Plain(start)),
                index,
            );
            let pointer = TypedPointer {
                base,
                index: index.expand.consume(),
                ty: PointerType {
                    pointee: base
                        .ty
                        .with_vector_size(self.slice.vector_size.unwrap_or(base.ty.vector_size())),
                    address_space: S::SPACE,
                    access: A::ACCESS,
                },
                checked: true,
                unroll_factor: 1,
            };
            pointer.validate().expect("valid typed pointer conversion");
            PtrExpand {
                node: PointerNode::Element(pointer),
                marker: PhantomData,
            }
        })
    }

    pub fn len(&self) -> usize {
        intrinsic!(|_scope| { self.slice.length })
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        intrinsic!(|scope| {
            let offset = add::expand(scope, self.slice.offset, start.clone());
            let length = sub::expand(scope, end, start);
            TypedSliceExpand {
                slice: SliceExpand {
                    offset,
                    length,
                    ..self.slice
                },
                marker: PhantomData,
            }
        })
    }
}
