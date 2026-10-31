use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn gather<S: Numeric + RudaElement, I: Int + RudaElement>(
    input: &Tensor<S>, input_imaginary: &Tensor<S>, indices: &Tensor<I>,
    output: &mut Tensor<S>, output_imaginary: &mut Tensor<S>,
    #[comptime] axis: usize, #[comptime] select: bool, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut remaining = p; let mut dim = output.rank(); let mut from = 0usize; let mut at = p;
        while dim > 0usize {
            dim -= 1usize;
            let coordinate = remaining % output.shape(dim); remaining /= output.shape(dim);
            if dim == axis {
                if comptime!(select) { at = coordinate; }
            } else { from += coordinate * input.stride(dim); }
        }
        let index = usize::cast_from(indices[address(indices, at)]);
        from += index * input.stride(axis);
        output[address(output, p)] = input[from];
        if comptime!(complex) { output_imaginary[address(output_imaginary, p)] = input_imaginary[from]; }
    }
}

#[ruda]
fn scale<S: Numeric, F: Float>(real: S, imaginary: S, alpha: S, alpha_imaginary: S, #[comptime] code: u32) -> (S, S) {
    let mut result = real; let mut result_i = imaginary;
    if comptime!(code == 3) { result = S::cast_from(real != S::cast_from(0u32) && alpha != S::cast_from(0u32)); }
    else if comptime!(code >= 4 && code <= 8) { result = S::cast_from(u64::cast_from(real) * u64::cast_from(alpha)); }
    else {
        let value = F::cast_from(real); let factor = F::cast_from(alpha);
        if comptime!(code == 10 || code == 11 || code == 14) {
            let value_i = F::cast_from(imaginary); let factor_i = F::cast_from(alpha_imaginary);
            result = S::cast_from(value * factor - value_i * factor_i);
            result_i = S::cast_from(value * factor_i + value_i * factor);
        } else { result = S::cast_from(value * factor); }
    }
    (result, result_i)
}

#[ruda]
fn add<S: Numeric, F: Float>(a: S, b: S, #[comptime] code: u32) -> S {
    let mut result = a;
    if comptime!(code == 3) { result = S::cast_from(a != S::cast_from(0u32) || b != S::cast_from(0u32)); }
    else if comptime!(code >= 4 && code <= 8) { result = S::cast_from(u64::cast_from(a) + u64::cast_from(b)); }
    else { result = S::cast_from(F::cast_from(a) + F::cast_from(b)); }
    result
}

#[ruda(launch)]
fn add_fibers<S: Numeric + RudaElement, F: Float, I: Int + RudaElement>(
    source: &Tensor<S>, source_imaginary: &Tensor<S>, indices: &Tensor<I>,
    target: &mut Tensor<S>, target_imaginary: &mut Tensor<S>, alpha: &Tensor<S>, alpha_imaginary: &Tensor<S>,
    rows: usize, #[comptime] axis: usize, #[comptime] select: bool, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < rows {
        let mut remaining = p; let mut dim = source.rank();
        let mut source_base = 0usize; let mut target_base = 0usize; let mut index_base = 0usize;
        while dim > 0usize {
            dim -= 1usize;
            if dim != axis {
                let coordinate = remaining % source.shape(dim); remaining /= source.shape(dim);
                source_base += coordinate * source.stride(dim); target_base += coordinate * target.stride(dim);
                if comptime!(!select) { index_base += coordinate * indices.stride(dim); }
            }
        }
        let mut factor = S::cast_from(1u32); let mut factor_i = S::cast_from(0u32);
        if comptime!(select) {
            factor = alpha[0usize];
            if comptime!(code == 10 || code == 11 || code == 14) { factor_i = alpha_imaginary[0usize]; }
        }
        let mut position = 0usize;
        while position < source.shape(axis) {
            let mut at = 0usize;
            if comptime!(select) { at = address(indices, position); }
            else { at = index_base + position * indices.stride(axis); }
            let index = usize::cast_from(indices[at]);
            let from = source_base + position * source.stride(axis); let to = target_base + index * target.stride(axis);
            let mut value = source[from]; let mut value_i = S::cast_from(0u32);
            if comptime!(code == 10 || code == 11 || code == 14) { value_i = source_imaginary[from]; }
            if comptime!(select) {
                let (r, i) = scale::<S, F>(value, value_i, factor, factor_i, code); value = r; value_i = i;
            }
            target[to] = add::<S, F>(target[to], value, code);
            if comptime!(code == 10 || code == 11 || code == 14) {
                target_imaginary[to] = add::<S, F>(target_imaginary[to], value_i, code);
            }
            position += 1usize;
        }
    }
}

fn grid(work: usize) -> RudaCount {
    assert!(work > 0 && work <= u32::MAX as usize, "indexed data operation exceeds U32 indexing");
    let blocks = work.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype, 10 | 11 | 14) {
        let result = scientific::component(value, imaginary);
        scientific::indexable(&result); result
    } else { View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype } }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 470 | 471)); assert_eq!(ints.len(), 2);
    assert!(ints[0] >= 0 && matches!(ints[1], 0 | 1));
    let axis = ints[0] as usize; let select = ints[1] != 0;
    assert_eq!(xs.len(), if op == 471 && select { 4 } else { 3 });
    let input = &xs[0]; let indices = &xs[1]; let output = &xs[2];
    assert_eq!(input.dtype, output.dtype); assert!(matches!(indices.dtype, 4 | 5));
    assert_eq!(input.shape.len(), output.shape.len()); assert!(axis < input.shape.len());
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 470 {
        assert!(input.dtype <= 14);
        if select {
            assert_eq!(indices.shape.len(), 1); assert_eq!(output.shape[axis], indices.len);
            assert!(output.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size == input.shape[dim]));
        } else {
            assert_eq!(output.shape, indices.shape);
            assert!(output.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size <= input.shape[dim]));
        }
        embedding_ops::validate_indices(&c, indices, input.shape[axis]);
        if output.len == 0 { return; }
        if select && super::graph_index::try_select(input,indices,output,axis) {return;}
        let re = part(input, false); let im = part(input, true); let out_re = part(output, false); let out_im = part(output, true);
        let complex = matches!(input.dtype, 10 | 11 | 14);
        macro_rules! launch { ($s:ty, $i:ty) => { unsafe { gather::launch::<$s, $i, CudaRuntime>(&c, grid(output.len), block,
            re.arg(), im.arg(), indices.arg(), out_re.arg(), out_im.arg(), axis, select, complex); }}; }
        macro_rules! data { ($s:ty) => { if indices.dtype == 4 { launch!($s, i64); } else { launch!($s, i32); } }; }
        match re.dtype {
            0 => data!(f32), 1 => data!(f16), 2 => data!(bf16), 3 | 8 | 12 | 13 => data!(u8),
            4 => data!(i64), 5 => data!(i32), 6 => data!(i16), 7 => data!(i8), 9 => data!(f64), _ => unreachable!(),
        }
    } else {
        assert!(input.dtype <= 11 || input.dtype == 14);
        assert!(input.dtype != 14 || select, "scatter_add does not dispatch complex32");
        if select {
            assert_eq!(indices.shape.len(), 1); assert_eq!(input.shape[axis], indices.len);
            assert!(input.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size == output.shape[dim]));
            assert_eq!(xs[3].dtype, input.dtype); assert_eq!(xs[3].len, 1);
        } else {
            assert_eq!(input.shape, indices.shape);
            assert!(input.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size <= output.shape[dim]));
        }
        embedding_ops::validate_indices(&c, indices, output.shape[axis]);
        if input.len == 0 { return; }
        if select && super::operation_scatter::try_scaled_index_add(input,indices,output,axis,Some(&xs[3])) {return;}
        let re = part(input, false); let im = part(input, true); let out_re = part(output, false); let out_im = part(output, true);
        let scalar = if select { &xs[3] } else { input };
        let alpha = part(scalar, false); let alpha_i = part(scalar, true);
        let rows = input.len / input.shape[axis];
        macro_rules! launch { ($s:ty, $f:ty, $i:ty) => { unsafe { add_fibers::launch::<$s, $f, $i, CudaRuntime>(&c, grid(rows), block,
            re.arg(), im.arg(), indices.arg(), out_re.arg(), out_im.arg(), alpha.arg(), alpha_i.arg(), rows, axis, select, input.dtype); }}; }
        macro_rules! data { ($s:ty, $f:ty) => { if indices.dtype == 4 { launch!($s, $f, i64); } else { launch!($s, $f, i32); } }; }
        match input.dtype {
            0 | 10 => data!(f32, f32), 1 | 14 => data!(f16, f32), 2 => data!(bf16, f32), 3 | 8 => data!(u8, f32),
            4 => data!(i64, f32), 5 => data!(i32, f32), 6 => data!(i16, f32), 7 => data!(i8, f32), 9 | 11 => data!(f64, f64), _ => unreachable!(),
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
