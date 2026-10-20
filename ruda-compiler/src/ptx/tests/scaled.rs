use super::*;

fn fp4() -> Type {
    Type::new(StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2))
}

fn matrix(ident: MatrixIdent) -> Matrix {
    Matrix {
        ident, m: 16, n: 8, k: 64,
        storage: if ident == MatrixIdent::Accumulator { FloatKind::F32.into() } else { fp4().storage_type() },
        layout: MatrixLayout::Undefined,
        scope: ruda_core::ir::MatrixScope::Plane,
    }
}

fn scaled_kernel(vector: bool) -> KernelDefinition {
    let mut k = kernel(if vector { "nvfp4_vectors" } else { "nvfp4_arrays" });
    k.ruda_dim = RudaDim::new_1d(32);
    let mut arguments = Vec::new();
    for (ty, count) in [(fp4(), 16), (fp4(), 8), (FloatKind::F32.into(), 4),
        (FloatKind::E4M3.into(), 4), (FloatKind::E4M3.into(), 4)]
    {
        let input = buffer(&mut k, ty.with_vector_size(count), Visibility::Read);
        let value = *k.body.create_local(ty.with_vector_size(count));
        load(&mut k.body, input, Variable::builtin(Builtin::UnitPos, UIntKind::U32.into()), value);
        if vector {
            arguments.push(value);
        } else {
            let array = *k.body.create_local_array(ty.with_vector_size(count), 1);
            store(&mut k.body, array, 0u32.into(), value);
            arguments.push(array);
        }
    }
    let ty = Type::from(FloatKind::F32).with_vector_size(4);
    let result = if vector { *k.body.create_local(ty) } else { *k.body.create_local_array(ty, 1) };
    k.body.register(Instruction::new(CoopMma::ExecuteScaled {
        matrix: matrix(MatrixIdent::A), registers_a: arguments[0], registers_b: arguments[1],
        registers_c: arguments[2], scales_a: arguments[3], scales_b: arguments[4], scales_factor: 4,
    }, result));
    let output = buffer(&mut k, ty, Visibility::ReadWrite);
    let value = if vector { result } else {
        let value = *k.body.create_local(ty);
        load(&mut k.body, result, 0u32.into(), value);
        value
    };
    store(&mut k.body, output, Variable::builtin(Builtin::UnitPos, UIntKind::U32.into()), value);
    k
}

fn compile_scaled(k: KernelDefinition) -> Result<PtxKernel> {
    PtxCompiler.compile(k, &PtxCompilationOptions { target: Some(PtxTarget { sm: 120, version: (8, 7) }) },
        ExecutionMode::Unchecked, UIntKind::U32.into())
}

#[test]
fn nvfp4_scaled_mma_emits_live_vector_and_array_paths() {
    for vector in [false, true] {
        let result = compile_scaled(scaled_kernel(vector)).unwrap();
        assert!(result.source.starts_with(".version 8.7\n.target sm_120a\n"));
        assert!(result.source.contains(".kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"));
        assert!(result.source.contains("st.global.f32"));
        assert!(result.source.contains("ld.global.u8"));
        export(&result);
    }
}

#[test]
fn nvfp4_scaled_mma_rejects_wrong_targets_and_formats() {
    for (sm, version) in [(89, (8, 7)), (100, (8, 7)), (120, (8, 6)), (121, (8, 7))] {
        let result = PtxCompiler.compile(scaled_kernel(true),
            &PtxCompilationOptions { target: Some(PtxTarget { sm, version }) },
            ExecutionMode::Unchecked, UIntKind::U32.into());
        assert!(matches!(result, Err(CompilationError::UnsupportedInstruction { .. })));
    }
    for change in 0..5 {
        let mut k = scaled_kernel(true);
        let op = k.body.instructions.iter_mut().find_map(|i| match &mut i.operation {
            Operation::CoopMma(CoopMma::ExecuteScaled { matrix, scales_factor, registers_a, registers_c, scales_a, .. }) =>
                Some((matrix, scales_factor, registers_a, registers_c, scales_a)),
            _ => None,
        }).unwrap();
        match change {
            0 => *op.1 = 2,
            1 => op.0.k = 32,
            2 => op.2.ty = Type::from(UIntKind::U8).with_vector_size(16),
            3 => op.3.ty = Type::from(FloatKind::TF32).with_vector_size(4),
            _ => op.4.ty = Type::from(FloatKind::E5M2).with_vector_size(4),
        }
        assert!(matches!(compile_scaled(k), Err(CompilationError::UnsupportedInstruction { .. })));
    }
}

#[test]
fn nvfp4_scaled_mma_checks_register_capacity() {
    let mut k = scaled_kernel(true);
    for i in &mut k.body.instructions {
        if let Operation::CoopMma(CoopMma::ExecuteScaled { registers_a, .. }) = &mut i.operation {
            registers_a.ty = fp4().with_vector_size(8);
        }
    }
    assert!(matches!(compile_scaled(k), Err(CompilationError::Validation { .. })));
}

#[test]
fn packed_fp4_transport_preserves_bits_without_changing_generic_target() {
    let mut k = kernel("packed_fp4_copy");
    let input = buffer(&mut k, fp4(), Visibility::Read);
    let output = buffer(&mut k, fp4(), Visibility::ReadWrite);
    let value = *k.body.create_local(fp4());
    let index = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
    load(&mut k.body, input, index, value);
    store(&mut k.body, output, index, value);
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(result.source.contains(".target sm_89\n"));
    assert!(result.source.contains("ld.global.u8"));
    assert!(result.source.contains("st.global.u8"));
    assert!(!result.source.contains("cvt.rn"));
    export(&result);
}

#[test]
fn packed_fp4_is_not_lowered_as_integer_arithmetic() {
    let mut k = kernel("packed_arithmetic");
    let x = Variable::constant(ConstantValue::Float(1.0), fp4());
    let out = *k.body.create_local(fp4());
    k.body.register(Instruction::new(Arithmetic::Add(BinaryOperator { lhs: x, rhs: x }), out));
    assert!(matches!(compile(k, ExecutionMode::Checked, UIntKind::U32), Err(CompilationError::UnsupportedInstruction { .. })));
}

#[test]
fn packed_fp4_constants_match_existing_cpp_splat_semantics() {
    for value in [-6.0, -0.0, 0.0, 0.5, 1.0, 1.5, 3.0, 6.0] {
        let bit = ruda_core::e2m1::from_f64(value).to_bits();
        assert_eq!(super::super::packed::constant(ConstantValue::Float(value)), (bit | bit << 4).to_string());
    }
}

#[test]
fn nvfp4_fragment_coordinates_compile_for_all_lanes_and_elements() {
    let mut k = kernel("nvfp4_fragment_coordinates");
    k.ruda_dim = RudaDim::new_1d(32);
    let output = buffer(&mut k, UIntKind::U32.into(), Visibility::ReadWrite);
    let lane = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
    let base = *k.body.create_local(UIntKind::U32.into());
    k.body.register(Instruction::new(Arithmetic::Mul(BinaryOperator { lhs: lane, rhs: 56u32.into() }), base));
    let mut position = 0u32;
    for (ident, count) in [(MatrixIdent::A, 16), (MatrixIdent::B, 8), (MatrixIdent::Accumulator, 4)] {
        for i in 0..count {
            for row in [true, false] {
                let coordinate = *k.body.create_local(UIntKind::U32.into());
                let op = if row { CoopMma::RowIndex { lane_id: lane, i: (i as u32).into(), matrix: matrix(ident) } }
                    else { CoopMma::ColIndex { lane_id: lane, i: (i as u32).into(), matrix: matrix(ident) } };
                k.body.register(Instruction::new(op, coordinate));
                let offset = *k.body.create_local(UIntKind::U32.into());
                k.body.register(Instruction::new(Arithmetic::Add(BinaryOperator { lhs: base, rhs: position.into() }), offset));
                store(&mut k.body, output, offset, coordinate);
                position += 1;
            }
        }
    }
    assert_eq!(position, 56);
    export(&compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap());
}
