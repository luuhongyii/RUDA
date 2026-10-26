use super::*;
use ruda_core::{ir::*, launch::RudaDim};

#[test]
fn typed_local_arrays_promote_constant_nonescaping_whole_elements_and_components() {
    for case in 0..9 {
        let mut scope = Scope::root(false);
        let ty = Type::from(FloatKind::F32).with_vector_size(4);
        let array = *scope.create_local_array(ty, 2);
        let mut pointer = TypedPointer {
            base: array, index: 0u32.into(),
            ty: PointerType { pointee: ty, address_space: AddressSpace::Local, access: PointerAccess::ReadWrite },
            checked: true, unroll_factor: 1,
        };
        if case == 1 { pointer.index = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into()); }
        if case == 2 { pointer.ty.pointee = FloatKind::F32.into(); }
        if case == 4 { pointer.index = 2u32.into(); }
        if case == 5 { pointer.ty.pointee = FloatKind::F32.into(); pointer.index = 8u32.into(); }
        if case == 6 { pointer.ty.pointee = UIntKind::U32.into(); }
        if case == 7 { pointer.unroll_factor = 2; }
        if case == 8 { pointer.ty.pointee = FloatKind::F32.into(); pointer.index = 5u32.into(); }
        let value = *scope.create_local(pointer.ty.pointee);
        scope.register(Instruction::new(Operator::Cast(UnaryOperator { input: 3f32.into() }), value));
        scope.register(Instruction::new(TypedMemory::Store { pointer, value }, array));
        scope.register(Synchronization::SyncRuda);
        if case == 3 {
            let address = *scope.create_local(UIntKind::U64.into());
            scope.register(Instruction::new(Operator::NativeAddress(BinaryOperator { lhs: array, rhs: 0u32.into() }), address));
            scope.register(Instruction::new(Operator::IndexAssign(IndexAssignOperator {
                index: 0u32.into(), value: address, vector_size: 0, unroll_factor: 1,
            }), Variable::new(VariableKind::GlobalOutputArray(1), UIntKind::U64.into())));
        }
        let result = *scope.create_local(pointer.ty.pointee);
        scope.register(Instruction::new(TypedMemory::Load(pointer), result));
        scope.register(Instruction::new(Operator::IndexAssign(IndexAssignOperator {
            index: 0u32.into(), value: result, vector_size: 0, unroll_factor: 1,
        }), Variable::new(VariableKind::GlobalOutputArray(0), pointer.ty.pointee)));
        let optimized = Optimizer::new(scope, RudaDim::new_1d(32), vec![], vec![]);
        let local_memory = optimized.node_ids().iter().any(|node| {
            optimized.block(*node).ops.borrow().values().any(|instruction| {
                matches!(&instruction.operation, Operation::TypedMemory(memory)
                    if matches!(memory.pointer().base.kind, VariableKind::LocalArray { .. }))
            })
        });
        assert_eq!(local_memory, !matches!(case, 0 | 2 | 8), "case={case}");
    }
}
