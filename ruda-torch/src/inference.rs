use super::*;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_qwen2_inference_api_version() -> u32 { 3 }

#[ruda(launch)]
pub(super) fn rms_storage_rounded<F: Float + RudaElement>(
    input: &Tensor<F>, weight: &Tensor<F>, output: &mut Tensor<F>, epsilon: f32,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = input.shape(input.rank() - 1);
    if row < input.len() / width {
        let base = row * width;
        let mut square = 0.0f32;
        let mut column = lane;
        while column < width {
            let value = f32::cast_from(input[base + column]);
            square += value * value;
            column += 32;
        }
        let inverse = (plane_sum(square) / width as f32 + epsilon).inverse_sqrt();
        let mut column = lane;
        while column < width {
            let normalized = F::cast_from(f32::cast_from(input[base + column]) * inverse);
            output[base + column] = F::cast_from(f32::cast_from(normalized) * f32::cast_from(weight[column]));
            column += 32;
        }
    }
}

#[ruda(launch)]
fn rotary_storage_rounded<F: Float + RudaElement>(
    input: &Tensor<F>, trig: &Tensor<F>, output: &mut Tensor<F>,
) {
    let position = ABSOLUTE_POS as usize;
    if position < output.len() {
        let width = input.shape(3);
        let token = (position / width) % input.shape(2);
        let head = (position / width / input.shape(2)) % input.shape(1);
        let batch = position / width / input.shape(2) / input.shape(1);
        let column = position % width;
        let base = batch * input.stride(0) + head * input.stride(1) + token * input.stride(2);
        let trig_position = (batch * input.shape(2) + token) * width + column;
        let mut partner = column + width / 2;
        if column >= width / 2 { partner = column - width / 2; }
        let mut rotated = f32::cast_from(input[base + partner * input.stride(3)]);
        if column < width / 2 { rotated = -rotated; }
        let first = F::cast_from(f32::cast_from(input[base + column * input.stride(3)]) * f32::cast_from(trig[trig_position]));
        let second = F::cast_from(rotated * f32::cast_from(trig[trig_position + trig.stride(0)]));
        output[position] = F::cast_from(f32::cast_from(first) + f32::cast_from(second));
    }
}

fn packed(view: &View) -> bool {
    let mut stride = 1usize;
    for (&size, &actual) in view.shape.iter().zip(&view.strides).rev() {
        if size > 1 && actual != stride { return false; }
        stride = stride.checked_mul(size).expect("inference shape overflow");
    }
    true
}

pub(super) fn launch(op: u32, input: &View, other: &View, output: &View, scalar: f32) {
    assert!(input.dtype <= 2 && input.dtype == other.dtype && input.dtype == output.dtype);
    if op == 111 {
        assert!(packed(input) && packed(output) && input.shape.len()==output.shape.len() && !output.shape.is_empty());
        let width=*output.shape.last().unwrap();assert!(width>0 && width<=u32::MAX as usize);
        let mut expected=output.shape.clone();*expected.last_mut().unwrap()=width.checked_mul(2).unwrap();
        assert_eq!(input.shape,expected);
        if output.len==0 {return;}
        let c=client();let grid=RudaCount::Static(output.len.div_ceil(128) as u32,1,1);
        macro_rules! run {($ty:ty)=>{unsafe {qwen_kernels::swiglu::launch::<$ty,CudaRuntime>(&c,grid,RudaDim::new_1d(128),
            primitives::tensor(input).into_array_arg(),primitives::tensor(output).into_array_arg(),width as u32)}};}
        match input.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);return;
    }
    assert_eq!(input.shape, output.shape);
    assert!(packed(output) && output.len <= u32::MAX as usize);
    if op == 109 {
        assert!(packed(input) && packed(other) && !input.shape.is_empty());
        let width = *input.shape.last().unwrap();
        assert!(width > 0 && scalar.is_finite() && scalar > 0.0);
        assert_eq!(other.shape.as_slice(), &[width]);
    } else {
        assert_eq!(op, 110);
        assert_eq!(input.shape.len(), 4);
        assert!(input.shape[3] > 0 && input.shape[3] % 2 == 0 && packed(other));
        assert_eq!(other.shape, vec![2, input.shape[0], input.shape[2], input.shape[3]]);
    }
    if output.len == 0 { return; }
    let client = client();
    let work = if op == 109 { output.len / input.shape.last().unwrap() * 32 } else { output.len };
    let count = u32::try_from(work.div_ceil(128)).expect("inference grid overflow");
    macro_rules! run {
        ($dtype:ty) => {
            if op == 109 {
                rms_storage_rounded::launch::<$dtype, CudaRuntime>(&client, RudaCount::Static(count, 1, 1),
                    RudaDim::new_1d(128), input.arg(), other.arg(), output.arg(), scalar);
            } else {
                rotary_storage_rounded::launch::<$dtype, CudaRuntime>(&client, RudaCount::Static(count, 1, 1),
                    RudaDim::new_1d(128), input.arg(), other.arg(), output.arg());
            }
        };
    }
    unsafe { match input.dtype { 0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() } }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if op == 109 { FUSED_RMS_NORM_CALLS.fetch_add(1, Ordering::Relaxed); }
}
