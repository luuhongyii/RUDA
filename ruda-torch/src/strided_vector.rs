use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

pub(super) fn storage(code: u32) -> StorageType {
    (match code {
        0 => ruda_core::tensor::DType::F32,
        1 => ruda_core::tensor::DType::F16,
        2 => ruda_core::tensor::DType::BF16,
        _ => unreachable!(),
    }).into()
}

pub(super) fn shape<R: Runtime>(dimensions: &[usize]) -> SequenceArg<R, FastDivmod<usize>> {
    let mut result = SequenceArg::new();
    for &dimension in dimensions { result.push(dimension); }
    result
}

#[ruda]
fn read_four<F: Float>(input: &Tensor<F>, index: usize, #[comptime] broadcast: bool) -> Vector<F, Const<4>> {
    let mut result = Vector::<F, Const<4>>::new(F::cast_from(0.0f32));
    if comptime!(broadcast) {
        result = Vector::new(input[index]);
    } else if index % 4 == 0 {
        let packed = input.slice(index, index + 4).into_vectorized().with_vector_size::<Const<4>>();
        result = packed[0];
    } else {
        #[unroll]
        for lane in 0usize..4 { result[lane] = input[index + lane]; }
    }
    result
}

#[ruda]
fn write_four<F: Float>(output: &mut Tensor<F>, index: usize, value: Vector<F, Const<4>>) {
    if index % 4 == 0 {
        let mut packed = output.slice_mut(index, index + 4).into_vectorized().with_vector_size::<Const<4>>();
        packed[0] = value;
    } else {
        #[unroll]
        for lane in 0usize..4 { output[index + lane] = value[lane]; }
    }
}

#[ruda(launch)]
pub(super) fn convert<I: Float, O: Float>(input: &Tensor<I>, output: &mut Tensor<O>,
    dimensions: &Sequence<FastDivmod<usize>>, #[comptime] broadcast: bool,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
) {
    let position = ABSOLUTE_POS as usize * 4;
    if position < output.len() {
        let mut remaining = position;let mut source = 0usize;let mut destination = 0usize;
        #[unroll]
        for index in 0usize..dimensions.len().comptime() {
            let dim = comptime!(dimensions.len() - index - 1);
            let (quotient, coordinate) = dimensions[dim].div_mod(remaining);
            remaining = quotient;
            source += coordinate * input.stride(dim);
            destination += coordinate * output.stride(dim);
        }
        let value = read_four(input, source, broadcast);
        write_four(output, destination, Vector::cast_from(value));
    }
}

#[ruda(launch)]
pub(super) fn cat<F: Float>(left: &Tensor<F>, right: &Tensor<F>, output: &mut Tensor<F>,
    dimensions: &Sequence<FastDivmod<usize>>, #[comptime] axis: usize,
    #[comptime] left_broadcast: bool, #[comptime] right_broadcast: bool, #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS as usize * 4;
    if position < output.len() {
        let mut remaining = position;let mut i = 0usize;let mut j = 0usize;let mut selected = true;
        #[unroll]
        for index in 0usize..dimensions.len().comptime() {
            let dim = comptime!(dimensions.len() - index - 1);
            let (quotient, coordinate) = dimensions[dim].div_mod(remaining);
            remaining = quotient;
            let mut second = coordinate;
            if comptime!(dim == axis) {
                if coordinate >= left.shape(dim) { selected = false;second -= left.shape(dim); }
            }
            i += coordinate * left.stride(dim);j += second * right.stride(dim);
        }
        let mut packed = output.slice_mut(position, position + 4).into_vectorized().with_vector_size::<Const<4>>();
        if selected { packed[0] = read_four(left, i, left_broadcast); }
        else { packed[0] = read_four(right, j, right_broadcast); }
    }
}
