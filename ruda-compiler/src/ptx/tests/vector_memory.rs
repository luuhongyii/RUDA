use super::*;

#[test]
fn packed_vector_memory_preserves_alignment_fallback_and_bounds() {
    for (kind, name, widths) in [
        (FloatKind::BF16, "bf16", &[2usize, 4, 8][..]),
        (FloatKind::F16, "f16", &[2usize, 4, 8][..]),
        (FloatKind::F32, "f32", &[2usize, 4][..]),
    ] {
        for &width in widths {
            for address in [UIntKind::U32, UIntKind::U64] {
                for (mode, checked) in [(ExecutionMode::Checked, "checked"), (ExecutionMode::Unchecked, "unchecked")] {
                    for shared in [false, true] {
                        if shared && mode == ExecutionMode::Checked { continue; }
                        let space = if shared { "shared_local" } else { "global" };
                        let mut k = kernel(&format!("packed_vector_memory_{name}_x{width}_{space}_{checked}_a{}", if address == UIntKind::U32 { 32 } else { 64 }));
                        let ty = Type::from(kind).with_vector_size(width);
                        let input = buffer(&mut k, ty, Visibility::Read);
                        let output = buffer(&mut k, ty, Visibility::ReadWrite);
                        let index = Variable::builtin(Builtin::UnitPos, address.into());
                        let mut value = *k.body.create_local(ty);
                        load(&mut k.body, input, index, value);
                        if shared {
                            let memory = *k.body.create_shared_array(ty, 64, Some(16));
                            store(&mut k.body, memory, index, value);
                            k.body.register(Instruction::no_out(Synchronization::SyncRuda));
                            let from_shared = *k.body.create_local(ty);
                            load(&mut k.body, memory, index, from_shared);
                            let local_array = *k.body.create_local_array(ty, 1);
                            store(&mut k.body, local_array, 0u32.into(), from_shared);
                            value = *k.body.create_local(ty);
                            load(&mut k.body, local_array, 0u32.into(), value);
                        }
                        store(&mut k.body, output, index, value);
                        let result = PtxCompiler.compile(k, &PtxCompilationOptions {
                            target: Some(PtxTarget { sm: 80, version: (8, 0) }),
                        }, mode, address.into()).unwrap();
                        let suffix = if kind == FloatKind::F32 { "f32" } else { "b32" };
                        let count = if kind == FloatKind::F32 { width } else { width / 2 };
                        let vector = if count == 1 { String::new() } else { format!("v{count}.") };
                        assert!(result.source.contains(&format!("ld.global.{vector}{suffix}")));
                        assert!(result.source.contains(&format!("st.global.{vector}{suffix}")));
                        assert!(result.source.contains("and.b64"));
                        assert!(result.source.contains("setp.ne.u64"));
                        assert!(result.source.contains(if kind == FloatKind::F32 { "ld.global.f32" } else { "ld.global.b16" }));
                        if mode == ExecutionMode::Checked { assert!(result.source.contains("setp.ge.")); }
                        if shared {
                            assert!(result.source.contains(&format!("ld.shared.{vector}{suffix}")));
                            assert!(result.source.contains(&format!("st.shared.{vector}{suffix}")));
                            assert!(!result.source.contains("ld.local.v"));
                            assert!(!result.source.contains("st.local.v"));
                            assert_eq!(result.source.matches("setp.ne.u64").count(), 2);
                        }
                        export(&result);
                    }
                }
            }
        }
    }
}

#[test]
fn underaligned_shared_vectors_keep_the_scalar_fallback() {
    let mut k = kernel("underaligned_shared_vectors");
    let ty = Type::from(FloatKind::BF16).with_vector_size(8);
    let input = buffer(&mut k, ty, Visibility::Read);
    let output = buffer(&mut k, ty, Visibility::ReadWrite);
    let index = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
    let value = *k.body.create_local(ty);
    let memory = *k.body.create_shared_array(ty, 64, Some(2));
    load(&mut k.body, input, index, value);
    store(&mut k.body, memory, index, value);
    k.body.register(Instruction::no_out(Synchronization::SyncRuda));
    let result = *k.body.create_local(ty);
    load(&mut k.body, memory, index, result);
    store(&mut k.body, output, index, result);
    let source = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap().source;
    assert!(source.contains("ld.shared.v4.b32"));
    assert!(source.contains("st.shared.v4.b32"));
    assert!(source.contains("ld.shared.b16"));
    assert!(source.contains("st.shared.b16"));
    assert_eq!(source.matches("setp.ne.u64").count(), 4);
}
