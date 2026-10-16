use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn clamp_float(value: f32, minimum: i64, maximum: i64) -> f32 {
    let low = f32::cast_from(minimum);
    let high = f32::cast_from(maximum);
    let mut result = value;
    if result != result || result < low { result = low; }
    if result > high { result = high; }
    result
}

#[ruda(launch)]
fn scalar_forward<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, output: &mut Tensor<S>, mask: &mut Tensor<u8>,
    scale: f32, inv_scale: f32, zero: i64, minimum: i64, maximum: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < input.len() {
        let x = F::cast_from(input[address(input, p)]);
        let raw = f32::cast_from((x * F::cast_from(inv_scale)).round() + F::cast_from(zero));
        let q = i64::cast_from(clamp_float(raw, minimum, maximum));
        output[address(output, p)] = S::cast_from(f32::cast_from(q - zero) * scale);
        mask[address(mask, p)] = u8::cast_from(raw >= f32::cast_from(minimum) && raw <= f32::cast_from(maximum));
    }
}

#[ruda(launch)]
fn tensor_forward<S: Float + RudaElement, F: Float + RudaElement, Z: Numeric + RudaElement>(
    input: &Tensor<S>, scales: &Tensor<f32>, zeros: &Tensor<Z>, enabled: &Tensor<i64>,
    output: &mut Tensor<S>, mask: &mut Tensor<u8>, minimum: i64, maximum: i64,
    #[comptime] inner: usize, #[comptime] channels: usize, #[comptime] mode: u32,
) {
    let p = ABSOLUTE_POS as usize;
    if p < input.len() {
        let x = input[address(input, p)];
        let mut active = true;
        if comptime!(mode == 0) { active = enabled[0] != 0; }
        if active {
            let mut channel = 0usize;
            if comptime!(mode >= 2) { channel = p / inner % channels; }
            let scale = scales[address(scales, channel)];
            let inv = 1.0f32 / scale;
            let zero = zeros[address(zeros, channel)];
            let scaled = F::cast_from(x) * F::cast_from(inv);
            let mut value = 0.0f32;
            let mut keep = false;
            if comptime!(mode == 1) {
                let raw = f32::cast_from(scaled.round() + F::cast_from(zero));
                let q = i64::cast_from(clamp_float(raw, minimum, maximum));
                value = f32::cast_from(q - i64::cast_from(zero)) * scale;
                keep = raw >= f32::cast_from(minimum) && raw <= f32::cast_from(maximum);
            } else {
                let mut q = 0i64;
                if comptime!(mode == 0) {
                    q = i64::cast_from(scaled.round() + F::cast_from(zero));
                } else if comptime!(mode == 2) {
                    q = i64::cast_from(scaled.round()) + i64::cast_from(zero);
                } else {
                    q = i64::cast_from((scaled + F::cast_from(zero)).round());
                }
                keep = q >= minimum && q <= maximum;
                if comptime!(mode == 2) {
                    let mut bounded = q;
                    if bounded < minimum { bounded = minimum; }
                    if bounded > maximum { bounded = maximum; }
                    value = f32::cast_from(bounded - i64::cast_from(zero)) * scale;
                } else {
                    value = (clamp_float(f32::cast_from(q), minimum, maximum) - f32::cast_from(zero)) * scale;
                }
            }
            output[address(output, p)] = S::cast_from(value);
            mask[address(mask, p)] = u8::cast_from(keep);
        } else {
            output[address(output, p)] = x;
            mask[address(mask, p)] = 1u8;
        }
    }
}

#[ruda(launch)]
fn learnable_backward(
    input: &Tensor<f32>, grad: &Tensor<f32>, scales: &Tensor<f32>, zeros: &Tensor<f32>,
    dx: &mut Tensor<f32>, ds: &mut Tensor<f32>, dz: &mut Tensor<f32>,
    minimum: i64, maximum: i64, factor: f32,
    #[comptime] inner: usize, #[comptime] channels: usize, #[comptime] per_channel: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < input.len() {
        let mut channel = 0usize;
        if comptime!(per_channel) { channel = p / inner % channels; }
        let scale = scales[address(scales, channel)];
        let inv = 1.0f32 / scale;
        let zp = zeros[address(zeros, channel)];
        let mut rounded = zp + 0.5f32;
        if comptime!(per_channel) { rounded = zp.round(); }
        // The learnable API's clamp preserves NaN, unlike fminf/fmaxf in scalar forward.
        if rounded < f32::cast_from(minimum) { rounded = f32::cast_from(minimum); }
        if rounded > f32::cast_from(maximum) { rounded = f32::cast_from(maximum); }
        let zero = i64::cast_from(rounded);
        let x = input[address(input, p)];
        let dy = grad[address(grad, p)];
        let q = i64::cast_from((x * inv).round() + f32::cast_from(zero));
        let inside = q >= minimum && q <= maximum;
        dx[address(dx, p)] = dy * f32::cast_from(inside);
        let mut small = f32::cast_from(minimum - zero);
        let mut big = f32::cast_from(maximum - zero);
        let mut bounded = q;
        if bounded < minimum { bounded = minimum; }
        if bounded > maximum { bounded = maximum; }
        let mut xfq = f32::cast_from(bounded - zero) * scale;
        if comptime!(per_channel) {
            small = f32::cast_from(minimum) - rounded;
            big = f32::cast_from(maximum) - rounded;
            xfq = (f32::cast_from(bounded) - rounded) * scale;
        }
        let mut dscale = 0.0f32;
        let mut dzero = 0.0f32;
        if inside {
            dscale = dy * (xfq - x) * inv * factor;
        } else {
            dzero = dy * -1.0f32 * scale * factor;
            if q < minimum { dscale = dy * small * factor; }
            else { dscale = dy * big * factor; }
        }
        ds[address(ds, p)] = dscale;
        dz[address(dz, p)] = dzero;
    }
}

#[ruda(launch)]
fn moving_observer<S: Float + RudaElement, F: Float + RudaElement>(
    observer_on: &Tensor<i64>, quant_on: &Tensor<i64>, current_min: &Tensor<S>, current_max: &Tensor<S>,
    running_min: &mut Tensor<S>, running_max: &mut Tensor<S>, scales: &mut Tensor<f32>, zeros: &mut Tensor<i32>,
    average: f32, minimum: i32, maximum: i32, #[comptime] symmetric: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < scales.len() {
        let lo_pos = address(running_min, p);
        let hi_pos = address(running_max, p);
        let mut lo = running_min[lo_pos];
        let mut hi = running_max[hi_pos];
        if observer_on[0] == 1 {
            let curr_lo = current_min[address(current_min, p)];
            let curr_hi = current_max[address(current_max, p)];
            let rate = S::cast_from(average);
            if f32::cast_from(lo).abs() == f32::new(f32::INFINITY) { lo = curr_lo; }
            else {
                let delta = S::cast_from(F::cast_from(curr_lo) - F::cast_from(lo));
                let update = S::cast_from(F::cast_from(rate) * F::cast_from(delta));
                lo = S::cast_from(F::cast_from(lo) + F::cast_from(update));
            }
            if f32::cast_from(hi).abs() == f32::new(f32::INFINITY) { hi = curr_hi; }
            else {
                let delta = S::cast_from(F::cast_from(curr_hi) - F::cast_from(hi));
                let update = S::cast_from(F::cast_from(rate) * F::cast_from(delta));
                hi = S::cast_from(F::cast_from(hi) + F::cast_from(update));
            }
            running_min[lo_pos] = lo;
            running_max[hi_pos] = hi;
        }
        if quant_on[0] == 1 {
            let mut low = f32::cast_from(lo);
            let mut high = f32::cast_from(hi);
            let range = maximum - minimum;
            if comptime!(symmetric) {
                if low < 0.0f32 && high > 0.0f32 {
                    let qlow = -(range / 2i32 + 1i32);
                    let qhigh = range / 2i32;
                    let left = (low / f32::cast_from(qlow)).abs();
                    let right = (high / f32::cast_from(qhigh)).abs();
                    let mut extent = left;
                    if left < right { extent = right; }
                    low = f32::cast_from(f64::cast_from(extent) * f64::cast_from(qlow));
                    high = f32::cast_from(f64::cast_from(extent) * f64::cast_from(qhigh));
                }
            }
            if low > 0.0f32 { low = 0.0f32; }
            if high < 0.0f32 { high = 0.0f32; }
            let mut scale = f32::cast_from((f64::cast_from(high) - f64::cast_from(low)) / f64::cast_from(range));
            if scale == 0.0f32 || (1.0f32 / scale).abs() == f32::new(f32::INFINITY) { scale = 0.1f32; }
            let low_ratio = f64::cast_from(low) / f64::cast_from(scale);
            let high_ratio = f64::cast_from(high) / f64::cast_from(scale);
            let low_error = f64::cast_from(minimum.abs()) + low_ratio.abs();
            let high_error = f64::cast_from(maximum.abs()) + high_ratio.abs();
            let mut initial = f64::cast_from(maximum) - high_ratio;
            if low_error < high_error { initial = f64::cast_from(minimum) - low_ratio; }
            if comptime!(symmetric) {
                if low < 0.0f32 && high > 0.0f32 { initial = f64::cast_from(minimum + maximum) / 2.0f64; }
            }
            let mut nudged = 0i32;
            if initial < f64::cast_from(minimum) { nudged = minimum; }
            else if initial > f64::cast_from(maximum) { nudged = maximum; }
            else { nudged = i32::cast_from(initial.round()); }
            scales[address(scales, p)] = scale;
            zeros[address(zeros, p)] = nudged;
        }
    }
}

fn grid(n: usize) -> RudaCount {
    RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("fake quantization launch overflow"), 1, 1)
}

fn geometry(input: &View, axis: i64) -> (usize, usize) {
    if axis < 0 { return (1, 1); }
    let axis = usize::try_from(axis).unwrap();
    assert!(axis < input.shape.len());
    (input.shape[axis + 1..].iter().product(), input.shape[axis])
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    let c = client();
    let block = RudaDim::new_1d(256);
    if op == 545 {
        assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 3); assert_eq!(values.len(), 1);
        assert_eq!(xs[0].shape, xs[1].shape); assert_eq!(xs[0].shape, xs[2].shape);
        assert_eq!(xs[0].dtype, xs[1].dtype); assert_eq!(xs[2].dtype, 3);
        assert!(ints[1] <= ints[0] && ints[0] <= ints[2]);
        if xs[0].len == 0 { return; }
        let scale = values[0] as f32;
        let inv = 1.0f32 / scale;
        macro_rules! run { ($s:ty,$f:ty) => { unsafe {
            scalar_forward::launch::<$s,$f,CudaRuntime>(&c,grid(xs[0].len),block,
                xs[0].arg(),xs[1].arg(),xs[2].arg(),scale,inv,ints[0],ints[1],ints[2]);
        } }; }
        match xs[0].dtype { 0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>panic!("unsupported fake quantization dtype") }
    } else if op == 546 {
        assert_eq!(xs.len(), 6); assert_eq!(ints.len(), 4); assert!(values.is_empty());
        let mode = u32::try_from(ints[3]).unwrap();
        assert!(mode <= 3 && ints[0] <= ints[1]);
        let (inner, channels) = geometry(&xs[0], if mode < 2 { -1 } else { ints[2] });
        assert_eq!(xs[1].dtype, 0); assert_eq!(xs[2].dtype, if mode == 3 { 0 } else { 4 });
        assert_eq!(xs[3].dtype, 4); assert_eq!(xs[4].dtype, xs[0].dtype); assert_eq!(xs[5].dtype, 3);
        assert_eq!(xs[0].shape, xs[4].shape); assert_eq!(xs[0].shape, xs[5].shape);
        assert_eq!(xs[1].len, channels); assert_eq!(xs[2].len, channels);
        assert!(mode != 0 || xs[3].len >= 1);
        if xs[0].len == 0 { return; }
        macro_rules! run { ($s:ty,$f:ty,$z:ty) => { unsafe {
            tensor_forward::launch::<$s,$f,$z,CudaRuntime>(&c,grid(xs[0].len),block,
                xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),
                ints[0],ints[1],inner,channels,mode);
        } }; }
        macro_rules! input { ($z:ty) => { match xs[0].dtype {
            0=>run!(f32,f32,$z),1=>run!(f16,f32,$z),2=>run!(bf16,f32,$z),9=>run!(f64,f64,$z),
            _=>panic!("unsupported fake quantization dtype")
        } }; }
        if mode == 3 { input!(f32); } else { input!(i64); }
    } else if op == 547 {
        assert_eq!(xs.len(), 7); assert_eq!(ints.len(), 3); assert_eq!(values.len(), 1);
        assert!(xs.iter().all(|v| v.dtype == 0)); assert!(ints[0] <= 0 && ints[1] >= 0);
        let (inner, channels) = geometry(&xs[0], ints[2]);
        assert_eq!(xs[2].len, channels); assert_eq!(xs[3].len, channels);
        for index in [1,4,5,6] { assert_eq!(xs[index].shape, xs[0].shape); }
        if xs[0].len == 0 { return; }
        unsafe { learnable_backward::launch::<CudaRuntime>(&c,grid(xs[0].len),block,
            xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),
            ints[0],ints[1],values[0] as f32,inner,channels,ints[2]>=0); }
    } else {
        assert_eq!(op, 548); assert_eq!(xs.len(), 8); assert_eq!(ints.len(), 3); assert_eq!(values.len(), 1);
        assert!(xs[0].dtype == 4 && xs[1].dtype == 4 && xs[0].len >= 1 && xs[1].len >= 1);
        assert_eq!(xs[6].dtype, 0); assert_eq!(xs[7].dtype, 5);
        for index in 2..=5 { assert_eq!(xs[index].dtype, xs[2].dtype); assert_eq!(xs[index].len, xs[6].len); }
        assert_eq!(xs[7].len, xs[6].len);
        if xs[6].len == 0 { return; }
        macro_rules! run { ($s:ty,$f:ty) => { unsafe {
            moving_observer::launch::<$s,$f,CudaRuntime>(&c,grid(xs[6].len),block,
                xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),
                values[0] as f32,ints[0] as i32,ints[1] as i32,ints[2]!=0);
        } }; }
        match xs[2].dtype { 0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>panic!("unsupported observer dtype") }
    }
}
