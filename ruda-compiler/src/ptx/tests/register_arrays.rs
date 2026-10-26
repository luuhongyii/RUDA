use super::*;

fn array(id: u32, ty: Type, length: usize) -> Variable {
    Variable::new(VariableKind::LocalArray { id, length, unroll_factor: 1 }, ty)
}

#[test]
fn manual_mma_arrays_remain_in_registers() {
    let mut k = kernel("mma_register_arrays");
    let bf16 = Type::scalar(ElemType::Float(FloatKind::BF16));
    let f32 = Type::scalar(ElemType::Float(FloatKind::F32));
    let a = array(10, bf16, 8);
    let b = array(11, bf16, 4);
    let c = array(12, f32, 4);
    for (v, count) in [(a, 8), (b, 4), (c, 4)] {
        for i in 0..count { store(&mut k.body, v, (i as u32).into(), Variable::constant(ConstantValue::Float(1.0), v.ty)); }
    }
    let matrix = Matrix { ident: MatrixIdent::Accumulator, m: 16, n: 8, k: 16,
        storage: f32.storage_type(), layout: MatrixLayout::RowMajor, scope: MatrixScope::Plane };
    k.body.register(Instruction::new(CoopMma::ExecuteManual { matrix, registers_a: a, registers_b: b, registers_c: c }, c));
    let value = local(13, f32);
    load(&mut k.body, c, 0u32.into(), value);
    let output = buffer(&mut k, f32, Visibility::ReadWrite);
    store(&mut k.body, output, 0u32.into(), value);
    let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(compiled.source.contains("mma.sync.aligned.m16n8k16"));
    assert!(!compiled.source.contains("ld.local"));
    assert!(!compiled.source.contains("st.local"));
    assert!(!compiled.source.contains(".local .align"));
    export(&compiled);
}

#[test]
fn dynamic_and_out_of_bounds_array_indices_keep_local_memory() {
    for index in [Variable::builtin(Builtin::UnitPosX, UIntKind::U32.into()), 4u32.into()] {
        let mut k = kernel("register_array_fallback");
        let ty = Type::scalar(ElemType::Float(FloatKind::F32));
        let a = array(10, ty, 4);
        store(&mut k.body, a, 0u32.into(), Variable::constant(ConstantValue::Float(1.0), ty));
        load(&mut k.body, a, index, local(11, ty));
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains("ld.local"));
        assert!(compiled.source.contains("st.local"));
        assert!(compiled.source.contains(".local .align"));
    }
}

#[test]
fn addressed_arrays_are_not_promoted() {
    let mut k = kernel("addressed_array");
    let ty = Type::scalar(ElemType::Float(FloatKind::F32));
    let a = array(10, ty, 4);
    store(&mut k.body, a, 0u32.into(), Variable::constant(ConstantValue::Float(1.0), ty));
    k.body.register(Instruction::new(Operator::NativeAddress(BinaryOperator { lhs: a, rhs: 0u32.into() }), local(11, Type::scalar(ElemType::UInt(UIntKind::U64)))));
    let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(compiled.source.contains(".local .align"));
}

#[test]
fn global_vector_memory_preserves_checked_and_misaligned_paths() {
    for (name, elem, width, instruction) in [
        ("vector_bf16x8", ElemType::Float(FloatKind::BF16), 8, "v4.b32"),
        ("vector_f32x4", ElemType::Float(FloatKind::F32), 4, "v4.f32"),
        ("vector_f64x2", ElemType::Float(FloatKind::F64), 2, "v2.f64"),
    ] {
        let mut k = kernel(name);
        let ty = Type::scalar(elem).with_vector_size(width);
        let input = buffer(&mut k, ty, Visibility::Read);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let value = local(10, ty);
        let index = Variable::builtin(Builtin::UnitPosX, UIntKind::U32.into());
        load(&mut k.body, input, index, value);
        store(&mut k.body, output, index, value);
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains(&format!("ld.global.{instruction}")));
        assert!(compiled.source.contains(&format!("st.global.{instruction}")));
        assert!(compiled.source.contains("and.b64"));
        assert!(compiled.source.contains("setp.ne.u64"));
        assert_eq!(compiled.source.matches("setp.ge.u32").count(), 2);
        export(&compiled);
    }
}

#[test]
fn half_matrix_fragments_keep_packed_registers() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let mut k = kernel("packed_mma_fragments");
        let half = Type::scalar(ElemType::Float(kind));
        let float = Type::scalar(ElemType::Float(FloatKind::F32));
        let shared = *k.body.create_shared_array(half, 512, Some(128));
        let a = array(10, half.with_vector_size(2), 4);
        let b = array(11, half.with_vector_size(2), 2);
        let c = array(12, float, 4);
        for (out, factor) in [(a, 4), (b, 2)] {
            k.body.register(Instruction::new(CoopMma::LoadMatrix {
                buffer: shared, offset: 0u32.into(), vector_size: Some(1), factor, transpose: false,
            }, out));
        }
        for index in 0u32..4 { store(&mut k.body, c, index.into(), Variable::constant(ConstantValue::Float(0.0), float)); }
        let matrix = Matrix { ident: MatrixIdent::Accumulator, m: 16, n: 8, k: 16,
            storage: float.storage_type(), layout: MatrixLayout::RowMajor, scope: MatrixScope::Plane };
        k.body.register(Instruction::new(CoopMma::ExecuteManual { matrix, registers_a: a, registers_b: b, registers_c: c }, c));
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains("ldmatrix.sync.aligned.m8n8.x4"));
        assert!(compiled.source.contains("mma.sync.aligned.m16n8k16"));
        assert!(!compiled.source.contains(".reg .b16 "));
        assert!(!compiled.source.contains(".local .align"));
    }
}

#[test]
fn paired_array_vector_access_is_packed() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let mut k = kernel("packed_vector_array");
        let ty = Type::scalar(ElemType::Float(kind)).with_vector_size(2);
        let a = array(10, ty, 4);
        let input = buffer(&mut k, ty, Visibility::Read);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let first = local(11, ty);let second = local(12, ty);
        load(&mut k.body, input, 0u32.into(), first);
        store(&mut k.body, a, 3u32.into(), first);
        load(&mut k.body, a, 3u32.into(), second);
        store(&mut k.body, output, 0u32.into(), second);
        assert!(crate::ptx::register_arrays::eligible(&k.body).1.contains(&10));
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains("mov.b32 {"));
        assert!(!compiled.source.contains(".local .align"));
    }
}

#[test]
fn scalar_array_access_disables_pair_packing() {
    let mut k = kernel("unpaired_vector_array");
    let scalar = Type::scalar(ElemType::Float(FloatKind::BF16));
    let a = array(10, scalar.with_vector_size(2), 4);
    store(&mut k.body, a, 0u32.into(), Variable::constant(ConstantValue::Float(1.0), a.ty));
    k.body.register(Instruction::new(Operator::Index(IndexOperator {
        list: a, index: 1u32.into(), vector_size: 1, unroll_factor: 1,
    }), local(11, scalar)));
    let (eligible, packed) = crate::ptx::register_arrays::eligible(&k.body);
    assert!(eligible.contains_key(&10));assert!(!packed.contains(&10));
    let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(!compiled.source.contains(".local .align"));
}
