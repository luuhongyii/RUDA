use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn pack<S: Numeric + RudaElement, D: Numeric + RudaElement>(
    source: &Tensor<S>, target: &mut Tensor<D>, columns: usize, inner: usize, inner_tiles: usize,
    #[comptime] reverse: bool, #[comptime] half: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < target.len() {
        if comptime!(reverse) {
            let column = p / inner; let k = p % inner;
            let packed = ((column / 16usize * inner_tiles + k / 16usize) * 16usize + k % 16usize) * 16usize + column % 16usize;
            target[address(target, p)] = D::cast_from(source[packed]);
        } else {
            let column = p / (inner_tiles * 256usize) * 16usize + p % 16usize;
            let k = p / 256usize % inner_tiles * 16usize + p / 16usize % 16usize;
            let mut value = 0.0f32;
            if column < columns && k < inner { value = f32::cast_from(source[address(source, column * inner + k)]); }
            if comptime!(half) {
                if value > 65504.0f32 { value = 65504.0f32; }
                if value < -65504.0f32 { value = -65504.0f32; }
            }
            target[p] = D::cast_from(value);
        }
    }
}

#[ruda]
fn float_input(value: f32, inverse_scale: f32, zero: i64) -> i32 {
    let mut q = (f32::cast_from(zero) + value * inverse_scale).round();
    if q < 0.0f32 { q = 0.0f32; }
    if q > 255.0f32 { q = 255.0f32; }
    i32::cast_from(q) - i32::cast_from(zero)
}

#[ruda(launch)]
fn integer_linear<I: Numeric + RudaElement, O: Numeric + RudaElement>(
    input: &Tensor<I>, weight: &Tensor<i8>, scales: &Tensor<f32>, zeros: &Tensor<i32>,
    bias: &Tensor<f32>, output: &mut Tensor<O>,
    rows: usize, columns: usize, inner: usize, column_tiles: usize, inner_tiles: usize, blocks: usize,
    input_zero: i64, output_zero: i64, input_scale: f32, output_scale: f32, slope: f32,
    minimum: i64, maximum: i64, #[comptime] float_source: bool,
    #[comptime] float_output: bool, #[comptime] activation: u32,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= blocks { terminate!(); }
    let tid = UNIT_POS as usize; let tx = tid % 16usize; let ty = tid / 16usize;
    let column_tile = block % column_tiles;
    let row = block / column_tiles * 16usize + ty; let column = column_tile * 16usize + tx;
    let mut weight_zero = 0i32; let mut weight_scale = 1.0f32;
    if column < columns {
        let mut qindex = 0usize; if scales.len() != 1usize { qindex = column; }
        weight_zero = zeros[address(zeros, qindex)]; weight_scale = scales[address(scales, qindex)];
    }
    let mut a_tile = SharedMemory::<i32>::new(256usize);
    let mut b_tile = SharedMemory::<i32>::new(256usize);
    let mut sum = 0u32;
    let inverse_scale = 1.0f32 / input_scale;
    for tile in 0..inner_tiles {
        let ak = tile * 16usize + tx; let bk = tile * 16usize + ty;
        let mut a = 0i32; let mut b = 0i32;
        if row < rows && ak < inner {
            let value = input[address(input, row * inner + ak)];
            if comptime!(float_source) { a = float_input(f32::cast_from(value), inverse_scale, input_zero); }
            else { a = i32::cast_from(value) - i32::cast_from(input_zero); }
        }
        if column < columns && bk < inner {
            b = i32::cast_from(weight[(column_tile * inner_tiles + tile) * 256usize + tid]) - weight_zero;
        }
        a_tile[tid] = a; b_tile[tid] = b;
        sync_ruda();
        for k in 0..16usize { sum += u32::cast_from(a_tile[ty * 16usize + k] * b_tile[k * 16usize + tx]); }
        sync_ruda();
    }
    if row < rows && column < columns {
        let acc = f32::cast_from(i32::cast_from(sum));
        let mut offset = 0.0f32; if bias.len() != 0usize { offset = bias[address(bias, column)]; }
        if comptime!(float_output) {
            let mut value = acc * input_scale * weight_scale + offset;
            if comptime!(activation == 1) { if value < 0.0f32 || value != value { value = 0.0f32; } }
            output[address(output, row * columns + column)] = O::cast_from(value);
        } else {
            let product_scale = input_scale * weight_scale;
            let mut scaled = 0.0f32;
            if comptime!(activation <= 1) {
                let mut corrected = acc;
                if bias.len() != 0usize { corrected += offset / product_scale; }
                scaled = corrected * (product_scale / output_scale);
            } else {
                let mut value = acc * input_scale * weight_scale + offset;
                if comptime!(activation == 2) { if value < 0.0f32 { value *= slope; } }
                else { value = value.tanh(); }
                scaled = value / output_scale;
            }
            let mut q = f64::cast_from(scaled.round()) + f64::cast_from(output_zero);
            if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
            if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
            if comptime!(activation == 1) { if q < f64::cast_from(output_zero) { q = f64::cast_from(output_zero); } }
            output[address(output, row * columns + column)] = O::cast_from(q);
        }
    }
}

#[ruda(launch)]
fn half_linear(
    input: &Tensor<f32>, weight: &Tensor<f16>, bias: &Tensor<f32>, output: &mut Tensor<f32>,
    rows: usize, columns: usize, inner: usize, column_tiles: usize, inner_tiles: usize, blocks: usize,
    #[comptime] relu: bool,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= blocks { terminate!(); }
    let tid = UNIT_POS as usize; let tx = tid % 16usize; let ty = tid / 16usize;
    let column_tile = block % column_tiles;
    let row = block / column_tiles * 16usize + ty; let column = column_tile * 16usize + tx;
    let mut a_tile = SharedMemory::<f32>::new(256usize);
    let mut b_tile = SharedMemory::<f32>::new(256usize);
    let mut sum = 0.0f32;
    for tile in 0..inner_tiles {
        let ak = tile * 16usize + tx; let bk = tile * 16usize + ty;
        let mut a = 0.0f32; let mut b = 0.0f32;
        if row < rows && ak < inner { a = input[address(input, row * inner + ak)]; }
        if column < columns && bk < inner { b = f32::cast_from(weight[(column_tile * inner_tiles + tile) * 256usize + tid]); }
        a_tile[tid] = a; b_tile[tid] = b;
        sync_ruda();
        for k in 0..16usize {
            if tile * 16usize + k < inner { sum += a_tile[ty * 16usize + k] * b_tile[k * 16usize + tx]; }
        }
        sync_ruda();
    }
    if row < rows && column < columns {
        if bias.len() != 0usize { sum += bias[address(bias, column)]; }
        if comptime!(relu) { if sum < 0.0f32 { sum = 0.0f32; } }
        output[address(output, row * columns + column)] = sum;
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((565..=567).contains(&op));
    let c = client();
    if op == 565 {
        assert_eq!(xs.len(), 2); assert_eq!(ints.len(), 3); assert!(values.is_empty());
        let n = usize::try_from(ints[0]).expect("negative linear output width");
        let k = usize::try_from(ints[1]).expect("negative linear input width");
        assert!((0..=1).contains(&ints[2])); let reverse = ints[2] != 0;
        let (plain, packed) = if reverse { (&xs[1], &xs[0]) } else { (&xs[0], &xs[1]) };
        assert_eq!(plain.shape, vec![n, k]);
        assert_eq!(packed.shape, vec![n.div_ceil(16), k.div_ceil(16), 16, 16]);
        if packed.len != 0 { assert_eq!(packed.strides, vec![k.div_ceil(16)*256, 256, 16, 1]); }
        assert!((plain.dtype == 7 && packed.dtype == 7) || (plain.dtype == 0 && packed.dtype == 1));
        if xs[1].len == 0 { return; }
        let grid = RudaCount::Static(u32::try_from(xs[1].len.div_ceil(256)).expect("linear pack launch overflow"), 1, 1);
        let block = RudaDim::new_1d(256);
        unsafe {
            if plain.dtype == 7 {
                pack::launch::<i8,i8,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),n,k,k.div_ceil(16),reverse,false);
            } else if reverse {
                pack::launch::<f16,f32,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),n,k,k.div_ceil(16),true,true);
            } else {
                pack::launch::<f32,f16,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),n,k,k.div_ceil(16),false,true);
            }
        }
        return;
    }
    assert_eq!(xs.len(), if op == 566 { 6 } else { 4 });
    assert_eq!(ints.len(), if op == 566 { 6 } else { 3 });
    assert_eq!(values.len(), if op == 566 { 3 } else { 0 });
    let (input, weight, bias, output) = (&xs[0], &xs[1], &xs[xs.len()-2], &xs[xs.len()-1]);
    let n = usize::try_from(ints[0]).expect("negative linear output width");
    let k = usize::try_from(ints[1]).expect("negative linear input width");
    assert!(input.shape.len() >= 2); assert_eq!(input.shape.last(), Some(&k));
    let m = input.shape[..input.shape.len()-1].iter().product::<usize>();
    let mut shape = input.shape.clone(); *shape.last_mut().unwrap() = n;
    assert_eq!(output.shape, shape);
    let nt = n.div_ceil(16); let kt = k.div_ceil(16);
    assert_eq!(weight.shape, vec![nt, kt, 16, 16]);
    if weight.len != 0 { assert_eq!(weight.strides, vec![kt*256, 256, 16, 1]); }
    assert_eq!(bias.dtype, 0); assert!(bias.len == 0 || bias.len == n);
    if op == 566 {
        assert_eq!(weight.dtype, 7); assert_eq!(xs[2].dtype, 0); assert_eq!(xs[3].dtype, 5);
        assert!(xs[2].len == 1 || xs[2].len == n); assert_eq!(xs[2].len, xs[3].len);
        assert!((0..=3).contains(&ints[4])); assert!((0..=1).contains(&ints[5]));
        assert_eq!(input.dtype == 0, ints[5] != 0);
        assert!((input.dtype == 0 && output.dtype == 0) || ((input.dtype == 7 || input.dtype == 8) && input.dtype == output.dtype));
        let (low, high) = if input.dtype == 7 { (-128,127) } else { (0,255) };
        assert!((low..=high).contains(&ints[2]));
        if output.dtype != 0 { assert!((low..=high).contains(&ints[3])); }
        else { assert!(ints[4] <= 1); }
    } else {
        assert_eq!(input.dtype, 0); assert_eq!(weight.dtype, 1); assert_eq!(output.dtype, 0);
        assert!((0..=1).contains(&ints[2]));
    }
    if output.len == 0 { return; }
    let blocks = m.div_ceil(16).checked_mul(nt).expect("linear launch overflow");
    assert!(blocks <= u32::MAX as usize);
    let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    if op == 567 {
        unsafe { half_linear::launch::<CudaRuntime>(&c,grid,block,input.arg(),weight.arg(),bias.arg(),output.arg(),
            m,n,k,nt,kt,blocks,ints[2]!=0); }
    } else {
        let (low, high) = if output.dtype == 7 { (-128,127) } else { (0,255) };
        macro_rules! run { ($i:ty,$o:ty) => { unsafe {
            integer_linear::launch::<$i,$o,CudaRuntime>(&c,grid,block,input.arg(),weight.arg(),xs[2].arg(),xs[3].arg(),
                bias.arg(),output.arg(),m,n,k,nt,kt,blocks,ints[2],ints[3],values[0] as f32,values[1] as f32,values[2] as f32,
                low,high,ints[5]!=0,output.dtype==0,ints[4] as u32);
        } }; }
        match input.dtype { 0=>run!(f32,f32),7=>run!(i8,i8),8=>run!(u8,u8),_=>unreachable!() }
    }
}
