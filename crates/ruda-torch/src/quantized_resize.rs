use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn nearest_index(position: usize, size: usize, scale: f32, #[comptime] exact: bool) -> usize {
    let mut coordinate = f32::cast_from(position) * scale;
    if comptime!(exact) {
        coordinate = f32::cast_from((f64::cast_from(position) + 0.5f64) * f64::cast_from(scale));
    }
    usize::cast_from(i64::min(i64::cast_from(coordinate.floor()), i64::cast_from(size) - 1i64))
}

#[ruda(launch)]
fn nearest<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, sd: f32, sh: f32, sw: f32,
    #[comptime] exact: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let ow = output.shape(4); let oh = output.shape(3); let od = output.shape(2);
        let w = p % ow; let h = p / ow % oh; let d = p / (ow * oh) % od;
        let plane = p / (ow * oh * od);
        let iw = input.shape(4); let ih = input.shape(3); let id = input.shape(2);
        let x = nearest_index(w, iw, sw, exact);
        let y = nearest_index(h, ih, sh, exact);
        let z = nearest_index(d, id, sd, exact);
        let source = ((plane * id + z) * ih + y) * iw + x;
        output[address(output, p)] = input[address(input, source)];
    }
}

#[ruda]
fn linear_coordinate(position: usize, size: usize, scale: f32, #[comptime] align: bool) -> (usize, usize, f32, f32) {
    let mut coordinate = scale * f32::cast_from(position);
    if comptime!(!align) {
        coordinate = scale * (f32::cast_from(position) + 0.5f32) - 0.5f32;
        if coordinate < 0.0f32 { coordinate = 0.0f32; }
    }
    let first = usize::cast_from(coordinate);
    let mut last = first; if first < size - 1usize { last += 1usize; }
    let weight = coordinate - f32::cast_from(first);
    (first, last, 1.0f32 - weight, weight)
}

#[ruda]
fn load<Q: Numeric + RudaElement>(input: &Tensor<Q>, plane: usize, y: usize, x: usize) -> f32 {
    f32::cast_from(input[address(input, (plane * input.shape(3) + y) * input.shape(4) + x)])
}

#[ruda(launch)]
fn bilinear<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, sh: f32, sw: f32, scale_ratio: f32,
    zero: i64, minimum: i64, maximum: i64, #[comptime] align: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let ow = output.shape(4); let oh = output.shape(3);
        let plane = p / (ow * oh); let y = p / ow % oh; let x = p % ow;
        let (y0, y1, wy0, wy1) = linear_coordinate(y, input.shape(3), sh, align);
        let (x0, x1, wx0, wx1) = linear_coordinate(x, input.shape(4), sw, align);
        let top = wx0 * load(input, plane, y0, x0) + wx1 * load(input, plane, y0, x1);
        let bottom = wx0 * load(input, plane, y1, x0) + wx1 * load(input, plane, y1, x1);
        let value = wy0 * top + wy1 * bottom - f32::cast_from(zero);
        let inverse_scale = 1.0f32 / scale_ratio;
        let mut q = f64::cast_from((f32::cast_from(zero) + value * inverse_scale).round());
        if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
        if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
        output[address(output, p)] = Q::cast_from(q);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 570 || op == 571); assert_eq!(xs.len(), 2);
    assert_eq!(ints.len(), if op == 570 { 1 } else { 2 }); assert_eq!(values.len(), 3);
    let input = &xs[0]; let output = &xs[1];
    assert_eq!(input.shape.len(), 5); assert_eq!(output.shape.len(), 5);
    assert_eq!(&input.shape[..2], &output.shape[..2]); assert_eq!(input.dtype, output.dtype);
    assert!(input.shape[1..].iter().all(|&n| n > 0));
    assert!((0..=1).contains(&ints[0]));
    let (minimum, maximum): (i64, i64) = match input.dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized resize requires int8, uint8 or int32 payload"),
    };
    if op == 571 {
        assert_eq!(input.shape[2], 1); assert_eq!(output.shape[2], 1);
        assert!((minimum..=maximum).contains(&ints[1]));
    }
    if output.len == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("quantized resize launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($q:ty) => { unsafe {
        if op == 570 {
            nearest::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                values[0] as f32,values[1] as f32,values[2] as f32,ints[0] != 0);
        } else {
            bilinear::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                values[0] as f32,values[1] as f32,values[2] as f32,ints[1],minimum,maximum,ints[0] != 0);
        }
    } }; }
    match input.dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
