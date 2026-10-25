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
