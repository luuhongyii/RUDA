use super::*;

#[test]
fn integer_remainder_uses_divisor_sign_without_floating_conversion() {
    for kind in [IntKind::I8, IntKind::I16, IntKind::I32, IntKind::I64] {
        let ty = Type::from(kind);
        let mut k = kernel(&format!("signed_remainder_{kind:?}"));
        let input = buffer(&mut k, ty, Visibility::Read);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let lhs = local(0, ty);
        let rhs = local(1, ty);
        let out = local(2, ty);
        load(&mut k.body, input, 0u32.into(), lhs);
        load(&mut k.body, input, 1u32.into(), rhs);
        k.body.register(Instruction::new(Arithmetic::Remainder(BinaryOperator { lhs, rhs }), out));
        store(&mut k.body, output, 0u32.into(), out);
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains("rem.s"));
        assert!(compiled.source.contains("xor.pred"));
        assert!(compiled.source.contains("and.pred"));
        assert!(!compiled.source.contains("cvt.f"));
        export(&compiled);
    }
}

#[test]
fn angles_and_reciprocal_hypot_cover_floating_storage() {
    for kind in [FloatKind::F16, FloatKind::BF16, FloatKind::F32, FloatKind::F64] {
        let ty = Type::scalar(ElemType::Float(kind));
        let mut k = kernel(&format!("angles_rhypot_{kind:?}"));
        let input = buffer(&mut k, ty, Visibility::Read);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let value = local(0, ty);
        load(&mut k.body, input, 0u32.into(), value);
        for (index, operation) in [
            Arithmetic::Degrees(UnaryOperator { input: value }),
            Arithmetic::Radians(UnaryOperator { input: value }),
            Arithmetic::Rhypot(BinaryOperator { lhs: value, rhs: value }),
        ].into_iter().enumerate() {
            let out = local(index as u32 + 1, ty);
            k.body.register(Instruction::new(operation, out));
            store(&mut k.body, output, (index as u32).into(), out);
        }
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains("sqrt.rn.f64"));
        export(&compiled);
    }
}

#[test]
fn shared_scalar_is_loaded_after_synchronization() {
    let mut k = kernel("shared_scalar_roundtrip");
    let ty = Type::from(UIntKind::U32);
    let output = buffer(&mut k, ty, Visibility::ReadWrite);
    let shared = *k.body.create_shared(ty);
    k.body.register(Instruction::new(Operation::Copy(7u32.into()), shared));
    k.body.register(Instruction::no_out(Synchronization::SyncRuda));
    k.body.register(Instruction::new(Operation::Copy(shared), local(0, ty)));
    store(&mut k.body, output, 0u32.into(), local(0, ty));
    let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert_eq!(compiled.shared_memory_bytes, 4);
    let store = compiled.source.find("st.shared.u32").unwrap();
    let barrier = compiled.source.find("bar.sync").unwrap();
    let load = compiled.source.find("ld.shared.u32").unwrap();
    assert!(store < barrier && barrier < load);
    export(&compiled);
}

#[test]
fn reinterpret_preserves_bytes_across_vector_widths() {
    for (from, to) in [
        (Type::from(IntKind::I8).with_vector_size(2), Type::from(FloatKind::F16)),
        (Type::from(FloatKind::F16), Type::from(IntKind::I8).with_vector_size(2)),
        (Type::from(UIntKind::U32).with_vector_size(2), Type::from(FloatKind::F64)),
        (Type::from(FloatKind::F64), Type::from(UIntKind::U32).with_vector_size(2)),
    ] {
        let mut k = kernel(&format!("recast_{}_{}", from.vector_size(), to.vector_size()));
        let input = buffer(&mut k, from, Visibility::Read);
        let output = buffer(&mut k, to, Visibility::ReadWrite);
        load(&mut k.body, input, 0u32.into(), local(0, from));
        k.body.register(Instruction::new(Operator::Reinterpret(UnaryOperator { input: local(0, from) }), local(1, to)));
        store(&mut k.body, output, 0u32.into(), local(1, to));
        export(&compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap());
    }
}

#[test]
fn atomic_vectors_and_interior_mutability() {
    for width in [1, 2, 4] {
        let mut k = kernel(&format!("atomic_interior_{width}"));
        let ty = Type::from(FloatKind::F16).with_vector_size(width);
        let atomic_ty = Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))).with_vector_size(width);
        let input = buffer(&mut k, atomic_ty, Visibility::Read);
        let pointer = local(0, atomic_ty);
        load(&mut k.body, input, 0u32.into(), pointer);
        let value = Variable::constant(ConstantValue::Float(5.0), ty);
        k.body.register(Instruction::new(AtomicOp::Add(BinaryOperator { lhs: pointer, rhs: value }), local(1, ty)));
        let compiled = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap();
        assert_eq!(compiled.source.matches("atom.add.noftz.f16").count(), width);
        export(&compiled);
    }
}
