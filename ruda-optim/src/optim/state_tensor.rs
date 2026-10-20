use ruda_model::{
    record::PrecisionSettings,
    tensor::{DType, Tensor, TensorCreationOptions, TensorData, backend::Backend},
};

pub(super) fn into_item<B: Backend, const D: usize, P: PrecisionSettings>(
    tensor: Tensor<B, D>,
) -> (DType, TensorData) {
    (tensor.dtype(), tensor.into_data().convert::<P::FloatElem>())
}

pub(super) fn from_item<B: Backend, const D: usize>(
    (dtype, data): (DType, TensorData),
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(
        data,
        TensorCreationOptions::new(device.clone()).with_dtype(dtype),
    )
}
