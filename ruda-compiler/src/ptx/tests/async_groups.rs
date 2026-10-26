use super::*;

#[test]
fn async_copy_groups_preserve_copy_wait_and_cta_visibility_order() {
    for address in [UIntKind::U32, UIntKind::U64] {
        for max_pending in [0u32, 1, 2, 7] {
            let mut k = kernel(&format!("async_groups_{address:?}_{max_pending}"));
            k.body.register_type::<usize>(address.into());
            let ty = Type::from(UIntKind::U32).with_vector_size(4);
            let input = buffer(&mut k, ty, Visibility::Read);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let shared = *k.body.create_shared_array(ty, 64, Some(16));
            let index = Variable::builtin(Builtin::UnitPos, address.into());
            k.body.register(Instruction::new(BarrierOps::CopyAsync {
                source: input, source_length: 64u32.into(), offset_source: index,
                offset_out: index, copy_length: 16, checked: false,
            }, shared));
            k.body.register(BarrierOps::CommitCopyGroup);
            k.body.register(BarrierOps::WaitCopyGroup { max_pending });
            k.body.register(BarrierOps::CommitCopyGroup);
            k.body.register(BarrierOps::WaitCopyGroup { max_pending: 0 });
            k.body.register(Synchronization::SyncRuda);
            let descriptor = PointerType { pointee: ty, address_space: AddressSpace::Shared,
                access: PointerAccess::ReadOnly };
            let pointer = *k.body.create_local(descriptor.value_type());
            let value = *k.body.create_local(ty);
            k.body.register(Instruction::new(PointerOp::Address(TypedPointer {
                base: shared, index, ty: descriptor, checked: false, unroll_factor: 1,
            }), pointer));
            k.body.register(Instruction::new(PointerOp::Load { pointer, checked: false }, value));
            store(&mut k.body, output, index, value);
            let compiled = compile(k, ExecutionMode::Unchecked, address).unwrap();
            assert_eq!(compiled.shared_memory_bytes, 1024);
            assert!(!compiled.source.contains("mbarrier"));
            assert_eq!(compiled.source.matches("cp.async.commit_group;").count(), 2);
            assert_eq!(compiled.source.matches("cp.async.wait_group ").count(), 2);
            let mut remaining = compiled.source.as_str();
            for instruction in ["cp.async.cg.shared.global", "cp.async.commit_group;",
                &format!("cp.async.wait_group {max_pending};"), "cp.async.commit_group;",
                "cp.async.wait_group 0;", "bar.sync 0;", "ld.shared.", "st.global."] {
                let position = remaining.find(instruction).unwrap_or_else(|| panic!("missing {instruction}: {}", compiled.source));
                remaining = &remaining[position + instruction.len()..];
            }
            export(&compiled);
        }
    }
}

#[test]
fn async_copy_groups_are_side_effecting_and_require_ampere() {
    for operation in [BarrierOps::CommitCopyGroup, BarrierOps::WaitCopyGroup { max_pending: 1 }] {
        assert!(!operation.is_pure());
        for target in [PtxTarget { sm: 75, version: (8, 0) }, PtxTarget { sm: 80, version: (6, 5) }] {
            let mut k = kernel("unsupported_async_group");
            k.body.register(operation.clone());
            assert!(matches!(PtxCompiler.compile(k, &PtxCompilationOptions { target: Some(target) },
                ExecutionMode::Unchecked, UIntKind::U32.into()), Err(CompilationError::UnsupportedInstruction { .. })));
        }
    }
}
