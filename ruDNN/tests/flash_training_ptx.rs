use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::{
    compiler::Compiler,
    ir::{FloatKind, ManagedVariable, Scope, Type, UIntKind, Variable, VariableKind},
    kernel::{KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg, Visibility},
    launch::{ExecutionMode, RudaDim},
};
use rudnn::flash_training::{forward_registers, register_backward::{key_value_registers, query_registers}};

fn buffer(kernel: &mut KernelDefinition, ty: Type, visibility: Visibility) -> ManagedVariable {
    let id = kernel.buffers.len() as u32;
    kernel.buffers.push(KernelArg { id, visibility, ty, size: None, has_extended_meta: false });
    ManagedVariable::Plain(Variable::new(if visibility == Visibility::Read {
        VariableKind::GlobalInputArray(id)
    } else { VariableKind::GlobalOutputArray(id) }, ty))
}

fn compile_kernel(name: &str, causal: bool, vectorized: bool) {
    let mut kernel = KernelDefinition {
        buffers: vec![], tensor_maps: vec![], scalars: vec![],
        ruda_dim: RudaDim::new_1d(if name == "forward" { 128 } else { 256 }),
        body: Scope::root(false),
        options: KernelOptions {
            kernel_name: format!("flash_register_{name}_{causal}_{vectorized}"),
            ..Default::default()
        },
    };
    kernel.body.register_type::<usize>(UIntKind::U32.into());
    let bf16 = Type::from(FloatKind::BF16);
    let fp32 = Type::from(FloatKind::F32);
    let q = buffer(&mut kernel, bf16, Visibility::Read);
    let k = buffer(&mut kernel, bf16, Visibility::Read);
    let v = buffer(&mut kernel, bf16, Visibility::Read);
    kernel.scalars.push(ScalarKernelArg { ty: UIntKind::U32.into(), count: 2 });
    kernel.scalars.push(ScalarKernelArg { ty: fp32.storage_type(), count: 1 });
    let heads = ManagedVariable::Plain(Variable::new(VariableKind::GlobalScalar(0), UIntKind::U32.into()));
    let kv_heads = ManagedVariable::Plain(Variable::new(VariableKind::GlobalScalar(1), UIntKind::U32.into()));
    let scale = ManagedVariable::Plain(Variable::new(VariableKind::GlobalScalar(0), fp32));
    if name == "forward" {
        let out = buffer(&mut kernel, bf16, Visibility::ReadWrite);
        let stats = buffer(&mut kernel, fp32, Visibility::ReadWrite);
        forward_registers::expand::<half::bf16>(&mut kernel.body,
            q.into(), k.into(), v.into(), out.into(), stats.into(),
            heads.into(), kv_heads.into(), scale.into(), causal, vectorized, bf16.storage_type());
    } else {
        let grad = buffer(&mut kernel, bf16, Visibility::Read);
        let stats = buffer(&mut kernel, fp32, Visibility::Read);
        let delta = buffer(&mut kernel, fp32, Visibility::Read);
        let out = buffer(&mut kernel, bf16, Visibility::ReadWrite);
        if name == "query" {
            query_registers::expand::<half::bf16>(&mut kernel.body,
                q.into(), k.into(), v.into(), grad.into(), stats.into(), delta.into(), out.into(),
                heads.into(), kv_heads.into(), scale.into(), causal, vectorized, bf16.storage_type());
        } else {
            let dv = buffer(&mut kernel, bf16, Visibility::ReadWrite);
            key_value_registers::expand::<half::bf16>(&mut kernel.body,
                q.into(), k.into(), v.into(), grad.into(), stats.into(), delta.into(), out.into(), dv.into(),
                heads.into(), kv_heads.into(), scale.into(), causal, vectorized, bf16.storage_type());
        }
    }
    let options = PtxCompilationOptions { target: Some(PtxTarget { version: (8, 0), sm: 80 }) };
    let compiled = PtxCompiler.compile(kernel, &options, ExecutionMode::Unchecked, UIntKind::U32.into())
        .unwrap_or_else(|error| panic!("{name}, causal={causal}, vectorized={vectorized}: {error:?}"));
    assert!(compiled.source.contains("mma.sync.aligned.m16n8k16"));
    assert!(!compiled.source.contains("ld.local."));
    assert!(!compiled.source.contains("st.local."));
    assert_eq!(compiled.shared_memory_bytes, if name == "forward" { 256 * 136 * 2 } else { 384 * 136 * 2 });
    if let Some(directory) = std::env::var_os("RUDA_FLASH_PTX_OUTPUT") {
        let path = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join(format!("{}.ptx", compiled.entrypoint)), compiled.source).unwrap();
    }
}

#[test]
fn register_flash_forward_lowers_to_sm80_ptx() {
    for causal in [false, true] {
        for vectorized in [false, true] { compile_kernel("forward", causal, vectorized); }
    }
}

#[test]
fn register_flash_query_backward_lowers_to_sm80_ptx() {
    for causal in [false, true] {
        for vectorized in [false, true] { compile_kernel("query", causal, vectorized); }
    }
}

#[test]
fn register_flash_key_value_backward_lowers_to_sm80_ptx() {
    for causal in [false, true] {
        for vectorized in [false, true] { compile_kernel("key_value", causal, vectorized); }
    }
}
