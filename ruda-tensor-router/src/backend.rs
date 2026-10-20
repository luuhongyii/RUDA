use super::{RouterTensor, RunnerChannel, RunnerClient, get_client};
use alloc::{format, string::String};
use ruda_tensor::{
    Backend, BackendTypes, DType, ExecutionError, QTensorPrimitive, quantization::QuantScheme,
};
use core::marker::PhantomData;

/// A backend that forwards the tensor operations to the appropriate backend (given multiple backends).
pub struct BackendRouter<R: RunnerChannel> {
    r: PhantomData<R>,
}

impl<R: RunnerChannel> core::fmt::Debug for BackendRouter<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("router"))
    }
}

impl<R: RunnerChannel> Clone for BackendRouter<R> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerChannel> Default for BackendRouter<R> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerClient> QTensorPrimitive for RouterTensor<R> {
    fn scheme(&self) -> &QuantScheme {
        if let DType::QFloat(scheme) = &self.dtype {
            scheme
        } else {
            // TODO: maybe `tensor.scheme()` should return an option
            panic!("Expected quantized float dtype, got {:?}", self.dtype)
        }
    }
}

impl<R: RunnerChannel> BackendTypes for BackendRouter<R> {
    type Device = R::Device;

    type FloatTensorPrimitive = RouterTensor<R::Client>;

    type FloatElem = R::FloatElem;

    type IntTensorPrimitive = RouterTensor<R::Client>;

    type IntElem = R::IntElem;

    type BoolTensorPrimitive = RouterTensor<R::Client>;

    type BoolElem = R::BoolElem;

    type QuantizedTensorPrimitive = RouterTensor<R::Client>;
}

impl<R: RunnerChannel> Backend for BackendRouter<R> {
    fn name(device: &Self::Device) -> String {
        format!("router<{}>", R::name(device))
    }

    fn seed(device: &Self::Device, seed: u64) {
        let client = get_client::<R>(device);
        client.seed(seed);
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = get_client::<R>(device);
        client.sync()
    }

    fn flush(device: &Self::Device) -> Result<(), ExecutionError> {
        get_client::<R>(device).flush()
    }

    fn profile<O: Send + 'static>(device: &Self::Device, options: ruda_tensor::ProfileOptions,
        func: impl FnOnce() -> O + Send) -> Result<(O, ruda_tensor::ProfileDuration), ExecutionError> {
        ruda_tensor::profile_with_tokens::<Self, O>(device, options, func)
    }
    fn profile_start(device: &Self::Device) -> Result<Option<ruda_tensor::ProfileToken>, ExecutionError> {
        get_client::<R>(device).profile_start()
    }
    fn profile_end(device: &Self::Device, token: ruda_tensor::ProfileToken, options: ruda_tensor::ProfileOptions)
        -> Result<ruda_tensor::ProfileDuration, ExecutionError> { get_client::<R>(device).profile_end(token, options) }
    fn profile_abandon(device: &Self::Device, token: ruda_tensor::ProfileToken) { get_client::<R>(device).profile_abandon(token); }

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        get_client::<R>(device).graph_prepare()
    }

    fn graph_start_capture(device: &Self::Device) -> Result<(), ExecutionError> {
        get_client::<R>(device).graph_start_capture()
    }

    fn graph_stop_capture(device: &Self::Device) -> Result<ruda_tensor::BackendGraph, ExecutionError> {
        get_client::<R>(device).graph_stop_capture()
    }

    fn graph_abort_capture(device: &Self::Device) {
        get_client::<R>(device).graph_abort_capture()
    }

    unsafe fn graph_replay(device: &Self::Device, graph: &ruda_tensor::BackendGraph) -> Result<(), ExecutionError> {
        unsafe { get_client::<R>(device).graph_replay(graph) }
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> ruda_tensor::DTypeUsageSet {
        let client = get_client::<R>(device);
        client.dtype_usage(dtype)
    }

    fn device_count(_: u16) -> usize {
        // This is what was there before, not sure if it's actually correct
        1
    }
}
