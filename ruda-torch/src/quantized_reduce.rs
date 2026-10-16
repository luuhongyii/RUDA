use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn moments<Q: Numeric + RudaElement, S: Numeric + RudaElement>(
    input: &Tensor<Q>, sums: &mut Tensor<i64>, squares: &mut Tensor<S>, #[comptime] deviation: bool,
) {
    let block = RUDA_POS_X as usize; let lane = UNIT_POS as usize;
    let tiles = sums.shape(1); let row = block / tiles; let tile = block % tiles;
    let width = input.shape(1);
    let mut sum = 0i64; let mut square = S::cast_from(0u32);
    for part in 0..4usize {
        let position = tile * 1024usize + lane + part * 256usize;
        if position < width {
            let value = i64::cast_from(input[address(input, row * width + position)]);
            sum += value;
            if comptime!(deviation) { let x = S::cast_from(value); square += x * x; }
        }
    }
    let mut sum_shared = SharedMemory::<i64>::new(256usize);
    let mut square_shared = SharedMemory::<S>::new(256usize);
    sum_shared[lane] = sum;
    if comptime!(deviation) { square_shared[lane] = square; }
    sync_ruda();
    let mut step = 128usize;
    while step > 0usize {
        if lane < step {
            let total = sum_shared[lane] + sum_shared[lane + step]; sum_shared[lane] = total;
            if comptime!(deviation) {
                let total_square = square_shared[lane] + square_shared[lane + step]; square_shared[lane] = total_square;
            }
        }
        sync_ruda(); step /= 2usize;
    }
    if lane == 0usize {
        sums[block] = sum_shared[0];
        if comptime!(deviation) { squares[block] = square_shared[0]; }
    }
}

#[ruda(launch)]
fn finish<Q: Numeric + RudaElement, S: Numeric + RudaElement>(
    sums: &Tensor<i64>, squares: &Tensor<S>, output: &mut Tensor<Q>, width: usize,
    scale: f64, correction: f64, zero: i64, minimum: i64, maximum: i64,
    #[comptime] deviation: bool,
) {
    let row = RUDA_POS_X as usize; let lane = UNIT_POS as usize; let tiles = sums.shape(1);
    let mut sum = 0i64; let mut square = S::cast_from(0u32);
    let mut tile = lane;
    while tile < tiles {
        sum += sums[row * tiles + tile];
        if comptime!(deviation) { square += squares[row * tiles + tile]; }
        tile += 256usize;
    }
    let mut sum_shared = SharedMemory::<i64>::new(256usize);
    let mut square_shared = SharedMemory::<S>::new(256usize);
    sum_shared[lane] = sum;
    if comptime!(deviation) { square_shared[lane] = square; }
    sync_ruda();
    let mut step = 128usize;
    while step > 0usize {
        if lane < step {
            let total = sum_shared[lane] + sum_shared[lane + step]; sum_shared[lane] = total;
            if comptime!(deviation) {
                let total_square = square_shared[lane] + square_shared[lane + step]; square_shared[lane] = total_square;
            }
        }
        sync_ruda(); step /= 2usize;
    }
    if lane == 0usize {
        if comptime!(deviation) {
            let n = f64::cast_from(width);
            let mut denominator = n - correction; if denominator < 0.0f64 { denominator = 0.0f64; }
            let sum_value = f64::cast_from(sum_shared[0]);
            let mean = sum_value / n; let unbiased_mean = sum_value / denominator;
            let mut variance = f64::cast_from(square_shared[0]) / denominator - 2.0f64 * mean * unbiased_mean
                + mean * mean * n / denominator;
            if variance < 0.0f64 { variance = 0.0f64; }
            let value = f32::cast_from(variance.sqrt() * scale);
            let inverse_scale = 1.0f32 / f32::cast_from(scale);
            let mut q = f64::cast_from((f32::cast_from(zero) + value * inverse_scale).round());
            if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
            if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
            output[address(output, row)] = Q::cast_from(q);
        } else {
            let value = f32::cast_from(sum_shared[0]) / f32::cast_from(width);
            output[address(output, row)] = Q::cast_from(value.round());
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 574 || op == 575); assert_eq!(xs.len(), 3);
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 574 {
        assert_eq!(ints.len(), 1); assert!(values.is_empty()); assert!((0..=1).contains(&ints[0]));
        let (input, sums, squares) = (&xs[0], &xs[1], &xs[2]);
        assert_eq!(input.shape.len(), 2); assert_eq!(sums.dtype, 4);
        let tiles = input.shape[1].div_ceil(1024).max(1);
        assert_eq!(sums.shape, vec![input.shape[0], tiles]); assert_eq!(sums.strides, vec![tiles,1]);
        assert_eq!(squares.dtype, if input.dtype == 5 { 0 } else { 4 });
        if ints[0] != 0 { assert_eq!(squares.shape, sums.shape); assert_eq!(squares.strides, sums.strides); }
        else { assert_eq!(squares.len, 0); }
        if sums.len == 0 { return; }
        let grid = RudaCount::Static(u32::try_from(sums.len).expect("quantized moments launch overflow"),1,1);
        macro_rules! run { ($q:ty,$s:ty) => { unsafe {
            moments::launch::<$q,$s,CudaRuntime>(&c,grid,block,input.arg(),sums.arg(),squares.arg(),ints[0] != 0);
        } }; }
        match input.dtype {5=>run!(i32,f32),7=>run!(i8,i64),8=>run!(u8,i64),_=>panic!("unsupported quantized moments payload")}
    } else {
        assert_eq!(ints.len(), 3); assert_eq!(values.len(), 2); assert!(ints[0] >= 0); assert!((0..=1).contains(&ints[2]));
        let (sums, squares, output) = (&xs[0], &xs[1], &xs[2]);
        assert_eq!(sums.dtype, 4); assert_eq!(sums.shape.len(), 2); assert_eq!(output.shape, vec![sums.shape[0]]);
        assert_eq!(sums.shape[1], (ints[0] as usize).div_ceil(1024).max(1));
        assert_eq!(sums.strides, vec![sums.shape[1],1]); assert_eq!(squares.dtype, if output.dtype == 5 { 0 } else { 4 });
        if ints[2] != 0 { assert_eq!(squares.shape, sums.shape); assert_eq!(squares.strides, sums.strides); }
        else { assert_eq!(squares.len, 0); }
        let (minimum, maximum): (i64, i64) = match output.dtype {
            5=>(i32::MIN as i64,i32::MAX as i64),7=>(-128,127),8=>(0,255),_=>panic!("unsupported quantized reduction payload"),
        };
        assert!((minimum..=maximum).contains(&ints[1]));
        if output.len == 0 { return; }
        let grid = RudaCount::Static(u32::try_from(output.len).expect("quantized reduction launch overflow"),1,1);
        macro_rules! run { ($q:ty,$s:ty) => { unsafe {
            finish::launch::<$q,$s,CudaRuntime>(&c,grid,block,sums.arg(),squares.arg(),output.arg(),
                ints[0] as usize,values[0],values[1],ints[1],minimum,maximum,ints[2] != 0);
        } }; }
        match output.dtype {5=>run!(i32,f32),7=>run!(i8,i64),8=>run!(u8,i64),_=>unreachable!()}
    }
}
