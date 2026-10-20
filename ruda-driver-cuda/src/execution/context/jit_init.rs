use crate::{compiler_backend::CompilerBackend, execution::nvrtc_program::NvrtcProgram};
use cudarc::driver::sys::*;
use std::{ffi::{CString, c_char, c_void}, ptr, sync::Once};

pub(super) fn initialize(backend: CompilerBackend, arch: u32) {
    static INITIALIZED: Once = Once::new();
    INITIALIZED.call_once(|| initialize_inner(backend, arch).expect("CUDA JIT initialization failed"));
}

struct LinkState(CUlinkState);

impl Drop for LinkState {
    fn drop(&mut self) {
        // SAFETY: This owner is created only after successful cuLinkCreate_v2.
        let status = unsafe { cuLinkDestroy(self.0) };
        if status != CUresult::CUDA_SUCCESS {
            log::warn!("CUDA JIT initialization link cleanup failed: {status:?}");
        }
    }
}

fn check(status: CUresult, operation: &str) -> Result<(), String> {
    if status == CUresult::CUDA_SUCCESS { Ok(()) }
    else { Err(format!("{operation}: {status:?}")) }
}

fn initialize_inner(backend: CompilerBackend, arch: u32) -> Result<(), String> {
    let mut ptx: Vec<c_char> = match backend {
        CompilerBackend::Nvrtc => {
            let source = CString::new(
                "extern \"C\" __global__ void ruda_jit_init(float* output, const float* input) {\n\
                 unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 float x = input[i];\n\
                 output[i] = logf(x * x + 1.0f);\n\
                 }\n"
            ).unwrap();
            let program = NvrtcProgram::new(source).map_err(|error| format!("{error:?}"))?;
            let target = if arch >= 90 { format!("--gpu-architecture=sm_{arch}a") }
                         else { format!("--gpu-architecture=sm_{arch}") };
            // SAFETY: The program owner remains alive through compilation and PTX extraction.
            unsafe {
                cudarc::nvrtc::result::compile_program(program.raw(), &[target.as_str(), "--fmad=false", "-lineinfo"])
                    .map_err(|error| format!("{error:?}"))?;
                cudarc::nvrtc::result::get_ptx(program.raw()).map_err(|error| format!("{error:?}"))?
            }
        }
        #[cfg(feature = "direct-ptx")]
        CompilerBackend::DirectPtx { version: (major, minor) } => {
            CString::new(format!(".version {major}.{minor}\n.target sm_{arch}\n.address_size 64\n.visible .entry ruda_jit_init() {{ ret; }}\n"))
                .unwrap().into_bytes_with_nul().into_iter().map(|byte| byte as c_char).collect()
        }
    };
    // Link PTX before returning the device to producers. Loading a cached module
    // alone need not initialize the JIT library or register its exit handlers.
    // No kernel is launched; the link state owns and releases the temporary cubin.
    unsafe {
        let mut raw = ptr::null_mut();
        check(cuLinkCreate_v2(0, ptr::null_mut(), ptr::null_mut(), &mut raw), "cuLinkCreate_v2")?;
        let link = LinkState(raw);
        check(cuLinkAddData_v2(link.0, CUjitInputType::CU_JIT_INPUT_PTX,
            ptx.as_mut_ptr() as *mut c_void, ptx.len(), c"ruda_jit_init".as_ptr(),
            0, ptr::null_mut(), ptr::null_mut()), "cuLinkAddData_v2")?;
        let mut cubin = ptr::null_mut();
        let mut size = 0;
        check(cuLinkComplete(link.0, &mut cubin, &mut size), "cuLinkComplete")?;
    }
    Ok(())
}
