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

/// A100 Qwen output/down projections: ordered serial split-K with BF16 writeback.
pub fn matmul_bf16_serial_k3<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>,
    out: RudaTensor<R>) -> Result<RudaTensor<R>, MatmulSetupError> {
    if a.dtype != DType::BF16 || b.dtype != DType::BF16 || out.dtype != DType::BF16
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims()!=2 || b.meta.num_dims()!=2 || out.meta.num_dims()!=2
        || a.meta.shape()[0]!=512 || !matches!(a.meta.shape()[1],3584|18944)
        || b.meta.shape()[..]!=[a.meta.shape()[1],3584]
        || out.meta.shape()[..]!=[512,3584] || out.meta.strides()[..]!=[3584,1]
        || a.meta.strides()[1]!=1 || (b.meta.strides()[0]!=1 && b.meta.strides()[1]!=1)
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new("unsupported BF16 serial-K3 projection".to_string())));
    }
    let k = a.meta.shape()[1];
    let block = k.div_ceil(3*64)*64;
    let partial = empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::from([3,512,3584]),DType::F32);
    for part in 0..3 {
        let start = part*block;
        let len = (k-start).min(block);
        let mut left = a.clone();
        left.meta.shape_mut()[1] = len;
        left.handle.offset_start = Some(a.handle.offset_start.unwrap_or(0)+(start*a.meta.strides()[1]*2) as u64);
        let mut right = b.clone();
        right.meta.shape_mut()[0] = len;
        right.handle.offset_start = Some(b.handle.offset_start.unwrap_or(0)+(start*b.meta.strides()[0]*2) as u64);
        let mut output = partial.clone();
        output.meta.remove(0);
        output.handle.offset_start = Some(partial.handle.offset_start.unwrap_or(0)+(part*512*3584*4) as u64);
        launch_matmul_with_precision(&Strategy::DoubleAsyncCyclicCmma(Default::default()),
            left,right,output,F32MathMode::Strict)?;
    }
    serial_reduce::launch::<R>(&a.client,RudaCount::Static((512*3584/128) as u32,1,1),RudaDim::new_1d(128),
        partial.into_array_arg(),out.clone().into_array_arg(),DType::BF16.into());
    Ok(out)
}

#[ruda(launch)]
fn serial_reduce<F: Float>(partial: &Array<f32>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let first = F::cast_from(partial[pos]);
        let second = F::cast_from(f32::cast_from(first)+partial[out.len()+pos]);
        out[pos] = F::cast_from(f32::cast_from(second)+partial[2*out.len()+pos]);
    }
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
