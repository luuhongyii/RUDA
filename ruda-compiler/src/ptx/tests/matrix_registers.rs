use super::*;

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
