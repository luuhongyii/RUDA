use super::*;
use alloc::{vec, vec::Vec};
use crate::dsl::frontend::LaunchMarker;
use ruda_core::ir::{AtomicOpCode, BinaryOperator, PointerOp, Select, UnaryOperator};

/// A first-class pointer value. Its address space and permissions are statically distinct.
///
/// ```compile_fail
/// use ruda_kernel::dsl::prelude::{GlobalSpace, ValuePtr};
/// fn write(pointer: ValuePtr<f32, GlobalSpace>) { pointer.store(1.0); }
/// ```
/// ```compile_fail
/// use ruda_kernel::dsl::prelude::{GlobalSpace, ValuePtr, Workgroup};
/// fn global(_: ValuePtr<f32, GlobalSpace>) {}
/// fn shared(pointer: ValuePtr<f32, Workgroup>) { global(pointer); }
/// ```
/// ```compile_fail
/// use ruda_kernel::dsl::prelude::{Atomic, GlobalSpace, ValuePtr};
/// fn mutate(pointer: ValuePtr<Atomic<u32>, GlobalSpace>) { pointer.fetch_add(1); }
/// ```
#[derive(Clone, Copy)]
pub struct ValuePtr<E: RudaPrimitive, S: PointerSpace, A: PointerPermission = ReadOnly> {
    marker: PhantomData<(E, S, A)>,
}

pub type ValuePtrExpand<E, S, A = ReadOnly> = NativeExpand<ValuePtr<E, S, A>>;

impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for ValuePtr<E, S, A> {
    type ExpandType = NativeExpand<Self>;
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for &ValuePtr<E, S, A> {
    type ExpandType = NativeExpand<ValuePtr<E, S, A>>;
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> RudaType for &mut ValuePtr<E, S, A> {
    type ExpandType = NativeExpand<ValuePtr<E, S, A>>;
}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> NativeAssign for ValuePtr<E, S, A> {}
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> Assign for ValuePtr<E, S, A> {
    fn expand_assign(&mut self, _: &mut Scope, value: Self) {
        *self = value;
    }
    fn init_mut(&self, _: &mut Scope) -> Self {
        *self
    }
}

fn emit(scope: &mut Scope, operation: PointerOp, ty: Type) -> ManagedVariable {
    let out = scope.create_local(ty);
    operation
        .validate(Some(*out))
        .expect("valid first-class pointer operation");
    scope.register(Instruction::new(operation, *out));
    out
}

#[ruda]
impl<E: RudaPrimitive, A: PointerPermission> ValuePtr<E, Workgroup, A> {
    /// Load after a workgroup barrier. Every unit must pass the same shared-memory address.
    pub fn workgroup_uniform_load(&self) -> E {
        intrinsic!(|scope| {
            let ty = self.expand.ty.pointer_type().expect("pointer value type").pointee();
            NativeExpand::new(emit(
                scope,
                PointerOp::UniformLoad { pointer: *self.expand },
                ty,
            ))
        })
    }
}

#[ruda]
impl<E: RudaPrimitive<Scalar: Numeric>, A: PointerPermission> ValuePtr<Atomic<E>, Workgroup, A> {
    /// Atomically load after a workgroup barrier; the address must be uniform across all units.
    pub fn workgroup_uniform_load_atomic(&self) -> E {
        intrinsic!(|scope| {
            let pointee = self.expand.ty.pointer_type().expect("pointer value type").pointee();
            let ty = Type::scalar(pointee.elem_type()).with_vector_size(pointee.vector_size());
            NativeExpand::new(emit(
                scope,
                PointerOp::UniformLoad { pointer: *self.expand },
                ty,
            ))
        })
    }
}

#[ruda]
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> ValuePtr<E, S, A> {
    pub fn offset(&self, offset: usize) -> Self {
        intrinsic!(|scope| {
            let ty = self.expand.ty;
            NativeExpand::new(emit(
                scope,
                PointerOp::Offset(BinaryOperator {
                    lhs: *self.expand,
                    rhs: *offset.expand,
                }),
                ty,
            ))
        })
    }

    pub fn read_only(&self) -> ValuePtr<E, S, ReadOnly> {
        intrinsic!(|scope| {
            let mut ty = self.expand.ty.pointer_type().expect("pointer value type");
            ty.access = PointerAccess::ReadOnly;
            NativeExpand::new(emit(
                scope,
                PointerOp::ReadOnly(UnaryOperator {
                    input: *self.expand,
                }),
                Type::Pointer(ty),
            ))
        })
    }

    pub fn select(&self, condition: bool, otherwise: Self) -> Self {
        intrinsic!(|scope| {
            let ty = self.expand.ty;
            NativeExpand::new(emit(
                scope,
                PointerOp::Select(Select {
                    cond: *condition.expand,
                    then: *self.expand,
                    or_else: *otherwise.expand,
                }),
                ty,
            ))
        })
    }

    pub fn load(&self) -> E {
        intrinsic!(|scope| {
            let ty = self
                .expand
                .ty
                .pointer_type()
                .expect("pointer value type")
                .pointee();
            NativeExpand::new(emit(
                scope,
                PointerOp::Load {
                    pointer: *self.expand,
                    checked: true,
                },
                ty,
            ))
        })
    }

    /// Export a byte address on backends that support native-address operations.
    pub fn native_address(&self) -> u64 {
        intrinsic!(|scope| {
            let ty = u64::as_type(scope);
            NativeExpand::new(emit(
                scope,
                PointerOp::NativeAddress(UnaryOperator {
                    input: *self.expand,
                }),
                ty,
            ))
        })
    }
}

#[ruda]
impl<E: RudaPrimitive, S: PointerSpace> ValuePtr<E, S, ReadWrite> {
    pub fn store(&self, value: E) {
        intrinsic!(|scope| {
            let op = PointerOp::Store {
                pointer: *self.expand,
                value: *value.expand,
                checked: true,
            };
            op.validate(None).expect("valid writable pointer store");
            scope.register(Instruction::no_out(op));
        })
    }
}

#[ruda]
impl<E: RudaPrimitive, S: PointerSpace, A: PointerPermission> ValuePtr<E, S, A> {
    /// Import a valid, aligned byte address in `S`. The allocation must outlive kernel execution;
    /// its address space, permissions and length are the caller's responsibility. Native backends only.
    pub fn from_native(address: u64) -> Self {
        intrinsic!(|scope| {
            let ty = PointerType {
                pointee: E::as_type(scope),
                address_space: S::SPACE,
                access: A::ACCESS,
            }
            .value_type();
            NativeExpand::new(emit(
                scope,
                PointerOp::FromNative {
                    address: *address.expand,
                    ty: ty.pointer_type().unwrap(),
                },
                ty,
            ))
        })
    }
}

fn atomic(
    scope: &mut Scope,
    pointer: Variable,
    operation: AtomicOpCode,
    values: Vec<Variable>,
    ty: Type,
) -> ManagedVariable {
    emit(
        scope,
        PointerOp::Atomic {
            pointer,
            operation,
            values,
        },
        ty,
    )
}

#[ruda]
impl<E: RudaPrimitive<Scalar: Numeric>, S: PointerSpace, A: PointerPermission>
    ValuePtr<Atomic<E>, S, A>
{
    pub fn atomic_load(&self) -> E {
        intrinsic!(|scope| {
            let ty = E::as_type(scope);
            NativeExpand::new(atomic(scope, *self.expand, AtomicOpCode::Load, vec![], ty))
        })
    }
}

macro_rules! numeric_atomic {
    ($name:ident, $opcode:ident) => {
        #[ruda]
        impl<E: RudaPrimitive<Scalar: Numeric>, S: PointerSpace> ValuePtr<Atomic<E>, S, ReadWrite> {
            pub fn $name(&self, value: E) -> E {
                intrinsic!(|scope| {
                    let ty = E::as_type(scope);
                    NativeExpand::new(atomic(
                        scope,
                        *self.expand,
                        AtomicOpCode::$opcode,
                        vec![*value.expand],
                        ty,
                    ))
                })
            }
        }
    };
}
numeric_atomic!(swap, Swap);
numeric_atomic!(fetch_add, Add);
numeric_atomic!(fetch_sub, Sub);
numeric_atomic!(fetch_min, Min);
numeric_atomic!(fetch_max, Max);

#[ruda]
impl<E: RudaPrimitive<Scalar: Numeric>, S: PointerSpace> ValuePtr<Atomic<E>, S, ReadWrite> {
    pub fn atomic_store(&self, value: E) {
        intrinsic!(|scope| {
            let op = PointerOp::Atomic {
                pointer: *self.expand,
                operation: AtomicOpCode::Store,
                values: vec![*value.expand],
            };
            op.validate(None).expect("valid writable atomic store");
            scope.register(Instruction::no_out(op));
        })
    }
}

macro_rules! integer_atomic {
    ($name:ident, $opcode:ident) => {
        #[ruda]
        impl<E: RudaPrimitive<Scalar: Int>, S: PointerSpace> ValuePtr<Atomic<E>, S, ReadWrite> {
            pub fn $name(&self, value: E) -> E {
                intrinsic!(|scope| {
                    let ty = E::as_type(scope);
                    NativeExpand::new(atomic(
                        scope,
                        *self.expand,
                        AtomicOpCode::$opcode,
                        vec![*value.expand],
                        ty,
                    ))
                })
            }
        }
    };
}
integer_atomic!(fetch_and, And);
integer_atomic!(fetch_or, Or);
integer_atomic!(fetch_xor, Xor);

#[ruda]
impl<E: RudaPrimitive<Scalar: Numeric>, S: PointerSpace> ValuePtr<Atomic<E>, S, ReadWrite> {
    pub fn compare_exchange_weak(&self, compare: E, value: E) -> E {
        intrinsic!(|scope| {
            let ty = E::as_type(scope);
            NativeExpand::new(atomic(
                scope,
                *self.expand,
                AtomicOpCode::CompareAndSwap,
                vec![*compare.expand, *value.expand],
                ty,
            ))
        })
    }
}

impl<E: RudaPrimitive, A: PointerPermission> LaunchArg for ValuePtr<E, GlobalSpace, A> {
    type RuntimeArg<R: LaunchMarker> = <Array<E> as LaunchArg>::RuntimeArg<R>;
    type CompilationArg = <Array<E> as LaunchArg>::CompilationArg;

    fn register<R: LaunchMarker>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        <Array<E> as LaunchArg>::register(arg, launcher)
    }
    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> Self::ExpandType {
        let array = if A::ACCESS == PointerAccess::ReadWrite {
            <Array<E> as LaunchArg>::expand_output(arg, builder)
        } else {
            <Array<E> as LaunchArg>::expand(arg, builder)
        };
        let ty = PointerType {
            pointee: array.expand.ty,
            address_space: AddressSpace::Global,
            access: A::ACCESS,
        };
        let index = usize::as_type(&builder.scope).constant(ruda_core::ir::ConstantValue::UInt(0));
        let op = PointerOp::Address(TypedPointer {
            base: *array.expand,
            index,
            ty,
            checked: true,
            unroll_factor: 1,
        });
        NativeExpand::new(emit(&mut builder.scope, op, ty.value_type()))
    }
    fn expand_output(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> Self::ExpandType {
        let array = <Array<E> as LaunchArg>::expand_output(arg, builder);
        let ty = PointerType {
            pointee: array.expand.ty,
            address_space: AddressSpace::Global,
            access: A::ACCESS,
        };
        let index = usize::as_type(&builder.scope).constant(ruda_core::ir::ConstantValue::UInt(0));
        NativeExpand::new(emit(
            &mut builder.scope,
            PointerOp::Address(TypedPointer {
                base: *array.expand,
                index,
                ty,
                checked: true,
                unroll_factor: 1,
            }),
            ty.value_type(),
        ))
    }
}
