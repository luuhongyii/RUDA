use ruda_core::{compiler::Compiler, ir::*, kernel::*, launch::*};

fn kernel(typed: bool, width: usize) -> KernelDefinition {
    let ty = Type::scalar(ElemType::Float(FloatKind::F32)).with_vector_size(width);
    let mut kernel = KernelDefinition {
        buffers: vec![
            KernelArg {
                id: 0,
                visibility: Visibility::Read,
                ty,
                size: None,
                has_extended_meta: false,
            },
            KernelArg {
                id: 1,
                visibility: Visibility::ReadWrite,
                ty,
                size: None,
                has_extended_meta: false,
            },
        ],
        tensor_maps: vec![],
        scalars: vec![],
        ruda_dim: RudaDim::new_1d(4),
        body: Scope::root(false),
        options: KernelOptions {
            kernel_name: "typed_pointer_copy".into(),
            ..Default::default()
        },
    };
    kernel.body.register_type::<usize>(UIntKind::U32.into());
    let input = Variable::new(VariableKind::GlobalInputArray(0), ty);
    let output = Variable::new(VariableKind::GlobalOutputArray(1), ty);
    let shared = *kernel.body.create_shared_array(ty, 4, None);
    let local = *kernel.body.create_local_array(ty, 4);
    let index = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
    let mut value = input;
    for (base, space, access, store) in [
        (input, AddressSpace::Global, PointerAccess::ReadOnly, false),
        (local, AddressSpace::Local, PointerAccess::ReadWrite, true),
        (local, AddressSpace::Local, PointerAccess::ReadOnly, false),
        (shared, AddressSpace::Shared, PointerAccess::ReadWrite, true),
        (shared, AddressSpace::Shared, PointerAccess::ReadOnly, false),
        (output, AddressSpace::Global, PointerAccess::ReadWrite, true),
    ] {
        if !store {
            value = *kernel.body.create_local(ty);
        }
        let pointer = TypedPointer {
            base,
            index,
            ty: PointerType {
                pointee: ty,
                address_space: space,
                access,
            },
            checked: true,
            unroll_factor: 1,
        };
        let memory = if store {
            TypedMemory::Store { pointer, value }
        } else {
            TypedMemory::Load(pointer)
        };
        let out = if store { base } else { value };
        let instruction = if typed {
            Instruction::new(memory, out)
        } else {
            memory.lower(Some(out)).unwrap()
        };
        kernel.body.register(instruction);
        if store && space == AddressSpace::Shared {
            kernel.body.register(Synchronization::SyncRuda);
        }
    }
    kernel
}

fn compare<C: Compiler + Default>(widths: &[usize]) {
    for width in widths {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let typed = C::default()
                .compile(
                    kernel(true, *width),
                    &Default::default(),
                    mode,
                    UIntKind::U32.into(),
                )
                .unwrap()
                .to_string();
            let legacy = C::default()
                .compile(
                    kernel(false, *width),
                    &Default::default(),
                    mode,
                    UIntKind::U32.into(),
                )
                .unwrap()
                .to_string();
            assert_eq!(
                typed, legacy,
                "typed and legacy code generation differ at width {width}, {mode:?}"
            );
        }
    }
}

#[cfg(feature = "lowering-cpp")]
#[test]
fn typed_pointer_cuda_hip_codegen() {
    use crate::dsl::lowering::cpp::{CudaCompiler, HipCompiler};
    compare::<CudaCompiler<ruda_compiler::cpp::cuda::mma::CudaWmmaCompiler>>(&[1, 4]);
    compare::<HipCompiler<ruda_compiler::cpp::hip::mma::RocWmmaCompiler>>(&[1, 4]);
}

#[cfg(feature = "lowering-metal")]
#[test]
fn typed_pointer_metal_codegen() {
    compare::<crate::dsl::lowering::cpp::MslCompiler>(&[1, 4]);
}

#[cfg(feature = "lowering-wgsl")]
#[test]
fn typed_pointer_wgsl_codegen() {
    compare::<crate::dsl::lowering::wgsl::WgslCompiler>(&[1, 4, 8]);
}

#[cfg(feature = "lowering-spirv")]
#[test]
fn typed_pointer_spirv_codegen() {
    for width in [1, 4, 8] {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut compiler =
                crate::dsl::lowering::spirv::compiler::<ruda_compiler::spirv::GLCompute>();
            let typed = compiler
                .compile(
                    kernel(true, width),
                    &Default::default(),
                    mode,
                    UIntKind::U32.into(),
                )
                .unwrap();
            let legacy = compiler
                .compile(
                    kernel(false, width),
                    &Default::default(),
                    mode,
                    UIntKind::U32.into(),
                )
                .unwrap();
            assert!(!typed.assembled_module.is_empty());
            assert_eq!(typed.bindings, legacy.bindings);
            assert_eq!(typed.shared_size, legacy.shared_size);
        }
    }
}

#[test]
fn typed_pointer_rejects_writable_read_only_binding() {
    let mut kernel = kernel(true, 1);
    kernel.buffers[1].visibility = Visibility::Read;
    assert!(kernel.validate_typed_memory().is_err());
}
