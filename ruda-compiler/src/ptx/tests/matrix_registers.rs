use super::*;

#[test]
fn manual_mma_accepts_constant_vector_sources_without_local_memory() {
    for (kind, accumulator) in [(FloatKind::F16, FloatKind::F16),
        (FloatKind::F16, FloatKind::F32), (FloatKind::BF16, FloatKind::F32)] {
        for initial in [0.0, 2.0] {
            let mut k = kernel(&format!("constant_register_mma_{kind:?}_{accumulator:?}_{}", initial as u32));
            k.ruda_dim = RudaDim::new_1d(32);
            let constant = |kind, width, value| Variable::new(
                VariableKind::Constant(ConstantValue::Float(value)), Type::from(kind).with_vector_size(width));
            let ty = Type::from(accumulator).with_vector_size(4);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let result = *k.body.create_local(ty);
            let matrix = Matrix { ident: MatrixIdent::A, m: 16, n: 8, k: 16,
                storage: kind.into(), layout: MatrixLayout::ColMajor, scope: MatrixScope::Plane };
            k.body.register(Instruction::new(CoopMma::ExecuteManual {
                matrix, registers_a: constant(kind, 8, 1.5), registers_b: constant(kind, 4, -0.5),
                registers_c: constant(accumulator, 4, initial),
            }, result));
            store(&mut k.body, output, Variable::builtin(Builtin::UnitPos, UIntKind::U32.into()), result);
            let options = PtxCompilationOptions { target: Some(PtxTarget { version: (8, 0), sm: 80 }) };
            let compiled = PtxCompiler.compile(k, &options, ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
            assert!(compiled.source.contains("mma.sync.aligned.m16n8k16"));
            assert!(!compiled.source.contains("ld.local."));
            assert!(!compiled.source.contains("st.local."));
            export(&compiled);
        }
    }
}

#[test]
fn manual_mma_rejects_undersized_constant_sources_and_constant_destination() {
    for operand in 0..4 {
        let mut k = kernel("invalid_constant_register_mma");
        let mut a = Variable::new(VariableKind::Constant(ConstantValue::Float(1.0)),
            Type::from(FloatKind::BF16).with_vector_size(8));
        let mut b = Variable::new(a.kind, a.ty.with_vector_size(4));
        let mut c = Variable::new(a.kind, Type::from(FloatKind::F32).with_vector_size(4));
        match operand { 0 => a.ty = a.ty.with_vector_size(7), 1 => b.ty = b.ty.with_vector_size(3),
            2 => c.ty = c.ty.with_vector_size(3), _ => {} }
        let out = if operand == 3 { c } else {
            *k.body.create_local(Type::from(FloatKind::F32).with_vector_size(4))
        };
        let matrix = Matrix { ident: MatrixIdent::A, m: 16, n: 8, k: 16,
            storage: FloatKind::BF16.into(), layout: MatrixLayout::ColMajor, scope: MatrixScope::Plane };
        k.body.register(Instruction::new(CoopMma::ExecuteManual {
            matrix, registers_a: a, registers_b: b, registers_c: c,
        }, out));
        let error = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap_err();
        let expected = if operand == 3 { "destination cannot be constant" } else { "container is too small" };
        assert!(format!("{error:?}").contains(expected), "{error:?}");
    }
}

#[test]
fn indexed_compound_assign_updates_borrowed_and_unborrowed_vector_components() {
    use ruda_kernel::dsl::prelude::{Const, NativeExpand, Vector, mul_assign_array_op};
    for mode in 0..3 {
        let mut k = kernel(&format!("compound_vector_component_{mode}"));
        k.ruda_dim = RudaDim::new_1d(1);
        k.body.register_type::<usize>(UIntKind::U32.into());
        let ty = Type::from(FloatKind::F32).with_vector_size(4);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let c = k.body.create_local(ty);
        k.body.register(Instruction::new(Operator::Cast(UnaryOperator { input: 3f32.into() }), *c));
        let reference = (mode != 0).then(|| k.body.reference_borrow(c.clone(), true).unwrap());
        let operand = if mode == 1 { reference.clone().unwrap() } else { c.clone() };
        let index = NativeExpand::from_lit(&k.body, 2u32);
        mul_assign_array_op::expand(&mut k.body,
            NativeExpand::<Vector<f32, Const<4>>>::new(operand), index, 4f32.into());
        let scalar = Type::from(FloatKind::F32);
        let product = k.body.instructions.iter().find(|instruction| {
            matches!(instruction.operation, Operation::Arithmetic(Arithmetic::Mul(_)))
        }).expect("component multiplication");
        assert_eq!(product.out.unwrap().ty, scalar);
        if mode != 0 {
            assert!(k.body.instructions.iter().any(|instruction| matches!(instruction.operation,
                Operation::Pointer(PointerOp::Store { value, .. }) if value == product.out.unwrap())));
        }
        let value = if let Some(reference) = reference {
            *k.body.reference_read(*reference).unwrap()
        } else { *c };
        store(&mut k.body, output, 0u32.into(), value);
        let compiled = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap();
        assert!(!compiled.source.contains("ld.local."), "mode={mode}\n{}", compiled.source);
        assert!(!compiled.source.contains("st.local."), "mode={mode}\n{}", compiled.source);
        export(&compiled);
    }
}

#[test]
fn borrowed_manual_mma_vectors_stay_in_registers_across_async_barriers_and_loops() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let mut k = kernel(&format!("borrowed_register_mma_{kind:?}"));
        k.body.register_type::<usize>(UIntKind::U32.into());
        let half = Type::from(kind);
        let accumulator = Type::from(FloatKind::F32).with_vector_size(4);
        let output = buffer(&mut k, accumulator, Visibility::ReadWrite);
        let shared = *k.body.create_shared_array(half.with_vector_size(8), 32, Some(16));
        let c = k.body.create_local(accumulator);
        k.body.register(Instruction::new(Operator::Cast(UnaryOperator { input: 0f32.into() }), *c));
        let c_reference = k.body.reference_borrow(c, true).unwrap();
        let barrier = *k.body.create_shared(OpaqueType::Barrier(BarrierLevel::Ruda));
        let elected = *k.body.create_local(Type::Scalar(ElemType::Bool.into()));
        k.body.register(Instruction::new(Comparison::Equal(BinaryOperator {
            lhs: Variable::builtin(Builtin::UnitPos, UIntKind::U32.into()), rhs: 0u32.into(),
        }), elected));
        k.body.register(BarrierOps::Init { barrier, is_elected: elected, arrival_count: 64u32.into() });
        let mut body = k.body.child();
        body.register(BarrierOps::CommitCopyAsync { barrier });
        body.register(BarrierOps::ArriveAndWait { barrier });
        let a = body.create_local(half.with_vector_size(8));
        let b = body.create_local(half.with_vector_size(4));
        for (registers, factor) in [(*a, 4), (*b, 2)] {
            body.register(Instruction::new(CoopMma::LoadMatrix {
                buffer: shared, offset: 0u32.into(), vector_size: Some(8), factor, transpose: false,
            }, registers));
        }
        let a_reference = body.reference_borrow(a, false).unwrap();
        let b_reference = body.reference_borrow(b, false).unwrap();
        let registers_a = *body.reference_read(*a_reference).unwrap();
        let registers_b = *body.reference_read(*b_reference).unwrap();
        let registers_c = *body.reference_read(*c_reference).unwrap();
        let result = *body.create_local(accumulator);
        let matrix = Matrix { ident: MatrixIdent::A, m: 16, n: 8, k: 16,
            storage: kind.into(), layout: MatrixLayout::ColMajor, scope: MatrixScope::Plane };
        body.register(Instruction::new(CoopMma::ExecuteManual { matrix, registers_a, registers_b, registers_c }, result));
        body.reference_write(*c_reference, result).unwrap();
        let i = *k.body.create_local_restricted(UIntKind::U32.into());
        k.body.register(Branch::RangeLoop(Box::new(RangeLoop {
            i, start: 0u32.into(), end: 3u32.into(), step: None, inclusive: false, scope: body,
        })));
        let value = *k.body.reference_read(*c_reference).unwrap();
        store(&mut k.body, output, 0u32.into(), value);
        let compiled = PtxCompiler.compile(k, &options(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
        assert!(compiled.source.contains("mma.sync.aligned.m16n8k16"));
        assert!(compiled.source.contains("cp.async.mbarrier.arrive.shared.b64"));
        assert!(!compiled.source.contains("ld.local."));
        assert!(!compiled.source.contains("st.local."));
        assert!(!compiled.source.contains(".local .align"));
        export(&compiled);
    }
}

#[test]
fn manual_mma_vector_operands_do_not_use_addressable_local_storage() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        for transpose in [false, true] {
            let mut k = kernel(&format!("register_operands_{kind:?}_{transpose}"));
            let ty = Type::from(kind);
            let shared = *k.body.create_shared_array(ty.with_vector_size(8), 32, Some(16));
            let a = *k.body.create_local(ty.with_vector_size(8));
            let b = *k.body.create_local(ty.with_vector_size(4));
            let c = *k.body.create_local(Type::from(FloatKind::F32).with_vector_size(4));
            k.body.register(Instruction::new(Operator::Cast(UnaryOperator { input: 0f32.into() }), c));
            let matrix = Matrix { ident: MatrixIdent::A, m: 16, n: 8, k: 16,
                storage: kind.into(), layout: MatrixLayout::ColMajor, scope: MatrixScope::Plane };
            for offset in [0u32, 16] {
                for (registers, factor) in [(a, 4), (b, 2)] {
                    k.body.register(Instruction::new(CoopMma::LoadMatrix {
                        buffer: shared, offset: offset.into(), vector_size: Some(8), factor, transpose,
                    }, registers));
                }
                k.body.register(Synchronization::SyncRuda);
                k.body.register(Instruction::new(CoopMma::ExecuteManual {
                    matrix, registers_a: a, registers_b: b, registers_c: c,
                }, c));
            }
            let result = PtxCompiler.compile(k, &options(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
            assert_eq!(result.source.matches("mma.sync.aligned.m16n8k16").count(), 2);
            assert_eq!(result.source.matches("ldmatrix.sync.aligned.m8n8.x4").count(), 2);
            assert_eq!(result.source.matches("ldmatrix.sync.aligned.m8n8.x2").count(), 2);
            assert!(!result.source.contains("ld.local."));
            assert!(!result.source.contains("st.local."));
            assert!(!result.source.contains(".local .align"));
            export(&result);
        }
    }
}

#[test]
fn manual_half_matrix_transfers_keep_aligned_register_words_packed() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        for width in [1, 2, 4, 8] {
            let mut k = kernel(&format!("packed_matrix_{kind:?}_{width}"));
            let ty = Type::from(kind);
            let shared = *k.body.create_shared_array(ty.with_vector_size(8), 32, Some(16));
            let a = *k.body.create_local_array(ty.with_vector_size(width), 8 / width);
            let b = *k.body.create_local_array(ty.with_vector_size(width), 4usize.div_ceil(width));
            for (registers, factor) in [(a, 4), (b, 2)] {
                k.body.register(Instruction::new(CoopMma::LoadMatrix {
                    buffer: shared, offset: 0u32.into(), vector_size: Some(8), factor, transpose: false,
                }, registers));
            }
            let c = *k.body.create_local_array(Type::from(FloatKind::F32).with_vector_size(4), 1);
            let matrix = Matrix { ident: MatrixIdent::A, m: 16, n: 8, k: 16,
                storage: kind.into(), layout: MatrixLayout::ColMajor, scope: MatrixScope::Plane };
            k.body.register(Instruction::new(CoopMma::ExecuteManual {
                matrix, registers_a: a, registers_b: b, registers_c: c,
            }, c));
            let result = PtxCompiler.compile(k, &options(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
            assert!(result.source.contains("mma.sync.aligned.m16n8k16"));
            assert!(result.source.contains("ldmatrix.sync.aligned.m8n8.x4"));
            if width >= 2 {
                assert!(!result.source.contains("ld.local.b16"));
                assert!(!result.source.contains("st.local.b16"));
                assert_eq!(result.source.matches("st.local.b32").count(), 10);
                assert_eq!(result.source.matches("ld.local.b32").count(), 10);
            } else {
                assert_eq!(result.source.matches("st.local.b16").count(), 12);
                assert_eq!(result.source.matches("ld.local.b16").count(), 12);
            }
            export(&result);
        }
    }
}

#[test]
fn manual_mma_vector_accumulator_is_not_addressable_local_storage() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let mut k = kernel(&format!("register_accumulator_{kind:?}"));
        let ty = Type::from(kind).with_vector_size(2);
        let a = *k.body.create_local_array(ty, 4);
        let b = *k.body.create_local_array(ty, 2);
        let c = *k.body.create_local(Type::from(FloatKind::F32).with_vector_size(4));
        k.body.register(Instruction::new(Operator::Cast(UnaryOperator { input: 0f32.into() }), c));
        let matrix = Matrix { ident: MatrixIdent::A, m: 16, n: 8, k: 16,
            storage: kind.into(), layout: MatrixLayout::ColMajor, scope: MatrixScope::Plane };
        for _ in 0..2 {
            k.body.register(Synchronization::SyncRuda);
            k.body.register(Instruction::new(CoopMma::ExecuteManual {
                matrix, registers_a: a, registers_b: b, registers_c: c,
            }, c));
        }
        let result = PtxCompiler.compile(k, &options(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
        assert_eq!(result.source.matches("mma.sync.aligned.m16n8k16").count(), 2);
        assert_eq!(result.source.matches("ld.local.b32").count(), 12);
        assert!(!result.source.contains("st.local."));
        assert_eq!(result.source.matches(".local .align").count(), 2);
        export(&result);
    }
}
