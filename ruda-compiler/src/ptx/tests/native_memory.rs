use super::*;

#[test]
fn unreachable_branch_emits_trap() {
    let mut k = kernel("unreachable_branch");
    k.body.register(Instruction::no_out(Branch::Unreachable));
    let ptx = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap().source;
    assert!(ptx.contains("trap;"));
}

#[test]
fn native_memory_preserves_load_store_and_space_conversion() {
    for address in [UIntKind::U32, UIntKind::U64] {
        for shared in [false, true] {
            let mut k = kernel("native_roundtrip");
            let ty: Type = UIntKind::U32.into();
            let input = buffer(&mut k, ty, Visibility::ReadWrite);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let storage = if shared { *k.body.create_shared_array(ty, 64, Some(16)) } else { input };
            let index = Variable::builtin(Builtin::UnitPos, address.into());
            let stored = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
            let pointer = *k.body.create_local(UIntKind::U64.into());
            let value = *k.body.create_local(ty);
            k.body.register(Instruction::new(Operator::NativeAddress(BinaryOperator { lhs: storage, rhs: index }), pointer));
            k.body.register(Instruction::no_out(Operator::NativeStore(BinaryOperator { lhs: pointer, rhs: stored })));
            k.body.register(Instruction::new(Operator::NativeLoad(UnaryOperator { input: pointer }), value));
            store(&mut k.body, output, index, value);
            let compiled = compile(k, ExecutionMode::Unchecked, address).unwrap();
            assert!(compiled.source.contains(if shared { "cvta.shared.u64" } else { "cvta.global.u64" }));
            assert!(compiled.source.contains("ld.u32"));
            assert!(compiled.source.contains("st.u32"));
        }
    }
}

#[test]
fn native_pointer_requires_u64() {
    let mut k = kernel("invalid_native_pointer");
    let pointer = local(0, UIntKind::U32.into());
    let value = local(1, UIntKind::U32.into());
    k.body.register(Instruction::new(Operator::NativeLoad(UnaryOperator { input: pointer }), value));
    assert!(compile(k, ExecutionMode::Unchecked, UIntKind::U32).is_err());
}

#[test]
fn pointer_value_native_roundtrip_and_offset() {
    let mut k = kernel("pointer_value_native");
    k.body.register_type::<usize>(UIntKind::U32.into());
    let ty: Type = UIntKind::U32.into();
    let input = buffer(&mut k, ty, Visibility::ReadWrite);
    let descriptor = PointerType { pointee: ty, address_space: AddressSpace::Global, access: PointerAccess::ReadWrite };
    let pointer = *k.body.create_local(descriptor.value_type());
    let address = *k.body.create_local(UIntKind::U64.into());
    let imported = *k.body.create_local(descriptor.value_type());
    let offset = *k.body.create_local(descriptor.value_type());
    let value = *k.body.create_local(ty);
    k.body.register(Instruction::new(PointerOp::Address(TypedPointer { base: input, index: 0u32.into(), ty: descriptor, checked: false, unroll_factor: 1 }), pointer));
    k.body.register(Instruction::new(PointerOp::NativeAddress(UnaryOperator { input: pointer }), address));
    k.body.register(Instruction::new(PointerOp::FromNative { address, ty: descriptor.value_type().pointer_type().unwrap() }, imported));
    k.body.register(Instruction::new(PointerOp::Offset(BinaryOperator { lhs: imported, rhs: 1u32.into() }), offset));
    k.body.register(Instruction::no_out(PointerOp::Store { pointer: offset, value: 7u32.into(), checked: false }));
    k.body.register(Instruction::new(PointerOp::Load { pointer: offset, checked: false }, value));
    store(&mut k.body, input, 0u32.into(), value);
    let ptx = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap().source;
    assert!(ptx.contains("cvta.global.u64") && ptx.contains("ld.u32") && ptx.contains("st.u32"));
    assert!(ptx.contains("mul.lo.u64") && ptx.contains("add.u64"));
}

#[test]
fn pointer_value_native_atomic_uses_existing_ptx_address_reference() {
    let mut k = kernel("pointer_value_native_atomic");
    k.body.register_type::<usize>(UIntKind::U32.into());
    let ty = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
    let input = buffer(&mut k, ty, Visibility::ReadWrite);
    let descriptor = PointerType { pointee: ty, address_space: AddressSpace::Global, access: PointerAccess::ReadWrite };
    let address = *k.body.create_local(UIntKind::U64.into());
    let pointer = *k.body.create_local(descriptor.value_type());
    let previous = *k.body.create_local(UIntKind::U32.into());
    k.body.register(Instruction::new(Operator::NativeAddress(BinaryOperator { lhs: input, rhs: 0u32.into() }), address));
    k.body.register(Instruction::new(PointerOp::FromNative { address, ty: descriptor.value_type().pointer_type().unwrap() }, pointer));
    k.body.register(Instruction::new(PointerOp::Atomic { pointer, operation: AtomicOpCode::Add, values: vec![1u32.into()] }, previous));
    let ptx = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap().source;
    assert!(ptx.contains("atom") && ptx.contains("add.u32"));
}
