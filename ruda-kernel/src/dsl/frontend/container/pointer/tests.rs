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

fn pointer_value_kernel(width: usize) -> KernelDefinition {
    let mut kernel = kernel(true, width);
    let mut instructions = Vec::new();
    for inst in core::mem::take(&mut kernel.body.instructions) {
        if let Operation::TypedMemory(memory) = &inst.operation {
            let pointer = *kernel.body.create_local(memory.pointer().ty.value_type());
            instructions.push(Instruction::new(PointerOp::Address(*memory.pointer()), pointer));
            instructions.push(match memory {
                TypedMemory::Load(_) => Instruction::new(PointerOp::Load { pointer, checked: true }, inst.out()),
                TypedMemory::Store { value, .. } => Instruction::no_out(PointerOp::Store { pointer, value: *value, checked: true }),
            });
        } else { instructions.push(inst); }
    }
    kernel.body.instructions = instructions;
    kernel
}

fn compile_values<C: Compiler + Default>(widths: &[usize]) {
    for width in widths {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let generated = C::default().compile(pointer_value_kernel(*width), &Default::default(), mode, UIntKind::U32.into()).unwrap().to_string();
            assert!(!generated.is_empty());
            assert!(!generated.contains("ptr<"));
        }
    }
}

#[cfg(feature = "lowering-cpp")]
#[test]
fn pointer_value_cuda_hip_codegen() {
    use crate::dsl::lowering::cpp::{CudaCompiler, HipCompiler};
    compile_values::<CudaCompiler<ruda_compiler::cpp::cuda::mma::CudaWmmaCompiler>>(&[1, 4]);
    compile_values::<HipCompiler<ruda_compiler::cpp::hip::mma::RocWmmaCompiler>>(&[1, 4]);
}

#[cfg(feature = "lowering-metal")]
#[test]
fn pointer_value_metal_codegen() { compile_values::<crate::dsl::lowering::cpp::MslCompiler>(&[1, 4]); }

#[cfg(feature = "lowering-wgsl")]
#[test]
fn pointer_value_wgsl_codegen() { compile_values::<crate::dsl::lowering::wgsl::WgslCompiler>(&[1, 4, 8]); }

#[cfg(feature = "lowering-spirv")]
#[test]
fn pointer_value_spirv_codegen() {
    for width in [1, 4, 8] {
        let result = crate::dsl::lowering::spirv::compiler::<ruda_compiler::spirv::GLCompute>()
            .compile(pointer_value_kernel(width), &Default::default(), ExecutionMode::Checked, UIntKind::U32.into()).unwrap();
        assert!(!result.assembled_module.is_empty());
    }
}

#[test]
fn pointer_value_binding_permissions() {
    let mut kernel = pointer_value_kernel(1);
    kernel.buffers[1].visibility = Visibility::Read;
    assert!(kernel.validate_typed_memory().is_err());
}

#[crate::dsl::prelude::ruda]
fn pointer_value_atomic_dsl(first: &mut crate::dsl::prelude::Array<crate::dsl::prelude::Atomic<u32>>) {
    use crate::dsl::prelude::*;
    let pointer = first.to_slice_mut().typed::<GlobalSpace>().ptr(0).value();
    pointer.atomic_store(1);
    let _loaded = pointer.read_only().atomic_load();
    let _old = pointer.swap(2);
    let _added = pointer.fetch_add(3);
    let _subtracted = pointer.fetch_sub(1);
    let _minimum = pointer.fetch_min(2);
    let _maximum = pointer.fetch_max(4);
    let _and = pointer.fetch_and(3);
    let _or = pointer.fetch_or(4);
    let _xor = pointer.fetch_xor(1);
    let _cas = pointer.compare_exchange_weak(4, 5);
}

fn pointer_value_atomic_kernel() -> KernelDefinition {
    use crate::dsl::prelude::{Array, Atomic, NativeExpand};
    let mut k = kernel(true, 1);
    k.body.instructions.clear();
    k.options.kernel_name = "pointer_value_atomic".into();
    let ty = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
    k.buffers = vec![KernelArg { id: 0, visibility: Visibility::ReadWrite, ty, size: None, has_extended_meta: false }];
    k.body.register_type::<u32>(UIntKind::U32.into());
    let array: NativeExpand<Array<Atomic<u32>>> = ManagedVariable::Plain(Variable::new(VariableKind::GlobalOutputArray(0), ty)).into();
    pointer_value_atomic_dsl::expand(&mut k.body, array);
    k
}

#[cfg(feature = "lowering-cpp")]
#[test]
fn pointer_value_atomic_cuda_hip_codegen() {
    use crate::dsl::lowering::cpp::{CudaCompiler, HipCompiler};
    let cuda = CudaCompiler::<ruda_compiler::cpp::cuda::mma::CudaWmmaCompiler>::default()
        .compile(pointer_value_atomic_kernel(), &Default::default(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap().to_string();
    let hip = HipCompiler::<ruda_compiler::cpp::hip::mma::RocWmmaCompiler>::default()
        .compile(pointer_value_atomic_kernel(), &Default::default(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap().to_string();
    assert!(cuda.contains("atomicAdd") && cuda.contains("atomicCAS"));
    assert!(hip.contains("atomicAdd") && hip.contains("atomicCAS"));
}

#[cfg(feature = "lowering-metal")]
#[test]
fn pointer_value_atomic_metal_codegen() {
    let metal = crate::dsl::lowering::cpp::MslCompiler::default()
        .compile(pointer_value_atomic_kernel(), &Default::default(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap().to_string();
    assert!(metal.contains("atomic_fetch_add") && metal.contains("atomic_compare_exchange"));
}

#[cfg(feature = "lowering-wgsl")]
#[test]
fn pointer_value_atomic_wgsl_codegen() {
    let wgsl = crate::dsl::lowering::wgsl::WgslCompiler::default()
        .compile(pointer_value_atomic_kernel(), &Default::default(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap().to_string();
    assert!(wgsl.contains("atomicAdd") && wgsl.contains("atomicCompareExchangeWeak"));
}

#[cfg(feature = "lowering-spirv")]
#[test]
fn pointer_value_atomic_spirv_codegen() {
    let result = crate::dsl::lowering::spirv::compiler::<ruda_compiler::spirv::GLCompute>()
        .compile(pointer_value_atomic_kernel(), &Default::default(), ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
    assert!(!result.assembled_module.is_empty());
}
