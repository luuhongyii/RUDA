use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::{RudaTensor,
    allocation::empty_device_contiguous_dtype, permutation::swap_dims}};
use crate::kernel_ir::{definition::MatmulSetupError, launch::Strategy};
use super::{F32MathMode, launch_matmul_with_precision};

/// A100 Qwen K/V projection: two interleaved 32-wide K streams, FP32 reduction.
pub fn matmul_bf16_sliced_k2<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>,
    out: RudaTensor<R>) -> Result<RudaTensor<R>, MatmulSetupError> {
    if a.dtype != DType::BF16 || b.dtype != DType::BF16 || out.dtype != DType::F32
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims()!=2 || b.meta.num_dims()!=2 || out.meta.num_dims()!=2
        || a.meta.shape()[..]!=[512,3584] || b.meta.shape()[..]!=[3584,512]
        || out.meta.shape()[..]!=[512,512] || out.meta.strides()[..]!=[512,1]
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("unsupported BF16 sliced-K2 projection".to_string())));
    }
    let left = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([2,512,1792]),DType::BF16);
    let right = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([2,512,1792]),DType::BF16);
    for (input,output,rhs) in [(a.clone(),left.clone(),false),(b,right.clone(),true)] {
        pack::launch::<R>(&a.client,RudaCount::Static((2*512*1792/128) as u32,1,1),RudaDim::new_1d(128),
            input.into_tensor_arg(),output.into_array_arg(),rhs,DType::BF16.into());
    }
    let partial = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([2,512,512]),DType::F32);
    launch_matmul_with_precision(&Strategy::DoubleAsyncCyclicCmma(Default::default()),
        left,swap_dims(right,1,2),partial.clone(),F32MathMode::Strict)?;
    reduce::launch::<R>(&a.client,RudaCount::Static((512*512/128) as u32,1,1),RudaDim::new_1d(128),
        partial.into_array_arg(),out.clone().into_array_arg());
    Ok(out)
}

#[ruda(launch)]
fn pack<F: Float>(input: &Tensor<F>, output: &mut Array<F>,
    #[comptime] rhs: bool, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() {
        let part = pos / (512*1792);
        let row = (pos % (512*1792)) / 1792;
        let local = pos % 1792;
        let inner = (local/32)*64+part*32+local%32;
        let mut index = row*input.stride(0)+inner*input.stride(1);
        if comptime!(rhs) { index = inner*input.stride(0)+row*input.stride(1); }
        output[pos] = input[index];
    }
}

#[ruda(launch)]
fn reduce(partial: &Array<f32>, out: &mut Array<f32>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() { out[pos] = partial[pos]+partial[512*512+pos]; }
}
