use crate::{DeviceRuntime, FloatElement, IntElement, element::BoolElement, RudaTensor};
use ruda_tensor::{
    Backend, BackendTypes, DTypeUsage, DTypeUsageSet, DeviceOps, ExecutionError, TensorData,
};
use ruda_core::tensor::DType;
use ruda_core::ir::features::{MmaConfig, TypeUsage};
use ruda::runtime::server::ComputeServer;
use std::marker::PhantomData;

#[cfg(not(feature = "fusion"))]
use ruda_tensor::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
#[cfg(not(feature = "fusion"))]
use ruda_tensor::graph::{BackendIr, TensorHandle};

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct DeviceBackend<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<BT>,
}

impl<R, F, I, BT> BackendTypes for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type Device = R::Device;

    type FloatElem = F;
    type IntElem = I;
    type BoolElem = BT;

    type FloatTensorPrimitive = RudaTensor<R>;
    type IntTensorPrimitive = RudaTensor<R>;
    type BoolTensorPrimitive = RudaTensor<R>;
    type QuantizedTensorPrimitive = RudaTensor<R>;
}

impl<R, F, I, BT> Backend for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn name(device: &Self::Device) -> String {
        let client = R::client(device);
        format!("ruda<{}>", R::name(&client))
    }

    fn seed(_device: &Self::Device, seed: u64) {
        rurand::seed(seed);
    }

    fn rng_state(device: &Self::Device) -> Result<alloc::vec::Vec<u8>, ExecutionError> {
        Self::sync(device)?;
        Ok(rurand::rng_state())
    }

    fn restore_rng_state(device: &Self::Device, state: &[u8]) -> Result<(), ExecutionError> {
        Self::sync(device)?;
        rurand::restore_rng_state(state)
            .map_err(|reason| ExecutionError::WithContext { reason: reason.into() })
    }

    fn ad_enabled(_device: &Self::Device) -> bool {
        false
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = R::client(device);
        futures_lite::future::block_on(client.sync()).map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })
    }

    fn flush(device: &Self::Device) -> Result<(), ExecutionError> {
        R::client(device).flush().map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })
    }

    fn profile<O: Send + 'static>(device: &Self::Device, options: ruda_tensor::ProfileOptions,
        func: impl FnOnce() -> O + Send) -> Result<(O, ruda_tensor::ProfileDuration), ExecutionError> {
        ruda_tensor::profile_with_tokens::<Self, O>(device, options, func)
    }

    fn profile_start(device: &Self::Device) -> Result<Option<ruda_tensor::ProfileToken>, ExecutionError> {
        R::client(device).profile_start().map(|window| Some(ruda_tensor::ProfileToken {
            id: window.token.id, opened_on: window.stream_id.value,
        })).map_err(|error| ExecutionError::WithContext { reason: format!("{error}") })
    }

    fn profile_end(device: &Self::Device, token: ruda_tensor::ProfileToken, _options: ruda_tensor::ProfileOptions)
        -> Result<ruda_tensor::ProfileDuration, ExecutionError> {
        R::client(device).profile_end(ruda::runtime::client::ProfileWindow {
            stream_id: ruda_tensor::StreamId { value: token.opened_on },
            token: ruda::runtime::server::ProfilingToken { id: token.id },
        }).map_err(|error| ExecutionError::WithContext { reason: format!("{error}") })
    }

    fn profile_abandon(device: &Self::Device, token: ruda_tensor::ProfileToken) {
        R::client(device).profile_abandon(ruda::runtime::client::ProfileWindow {
            stream_id: ruda_tensor::StreamId { value: token.opened_on },
            token: ruda::runtime::server::ProfilingToken { id: token.id },
        });
    }

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        R::client(device).graph_prepare().map_err(|err| ExecutionError::WithContext { reason: format!("{err}") })
    }

    fn graph_start_capture(device: &Self::Device) -> Result<(), ExecutionError> {
        R::client(device).graph_start_capture().map_err(|err| ExecutionError::WithContext { reason: format!("{err}") })
    }

    fn graph_stop_capture(device: &Self::Device) -> Result<ruda_tensor::BackendGraph, ExecutionError> {
        R::client(device).graph_stop_capture().map(|graph| ruda_tensor::BackendGraph::new(DeviceGraph(graph)))
            .map_err(|err| ExecutionError::WithContext { reason: format!("{err}") })
    }

    fn graph_abort_capture(device: &Self::Device) { R::client(device).graph_abort_capture(); }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        let client = R::client(device);
        client.memory_persistent_allocation(input, func).unwrap()
    }

    fn memory_cleanup(device: &Self::Device) {
        let client = R::client(device);
        client.memory_cleanup();
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        let client = R::client(device);
        client.staging(data.map(|td| &mut td.bytes), false);
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        ruda_kernel::tensor::capability::supports_dtype::<R>(device, dtype)
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        let client = R::client(device);

        let props = client.properties();
        let storage = dtype.into();
        let usage = props.type_usage(storage);

        let mut out = DTypeUsageSet::new();

        if usage.is_superset(TypeUsage::Buffer | TypeUsage::Conversion) {
            out |= DTypeUsage::Storage;
        }

        if usage.contains(TypeUsage::Arithmetic) {
            out |= DTypeUsage::Arithmetic;
        }

        let has_mma = |cfg: &MmaConfig| {
            cfg.a_type == storage || cfg.b_type == storage || cfg.cd_type == storage
        };
        if props.features.matmul.cmma.iter().any(has_mma)
            || props.features.matmul.mma.iter().any(has_mma)
        {
            out |= DTypeUsage::Accelerated;
        }

        out
    }

    fn device_count(type_id: u16) -> usize {
        let client = R::client(&Default::default());
        client.device_count(type_id)
    }
}

impl<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> core::fmt::Debug
    for DeviceBackend<R, F, I, BT>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RudaBackend")
    }
}

struct DeviceGraph<R: DeviceRuntime>(ruda::runtime::capture::RuntimeGraph<R>);

impl<R: DeviceRuntime> ruda_tensor::HardwareGraph for DeviceGraph<R> {
    unsafe fn replay(&self) -> Result<(), ExecutionError> {
        unsafe { self.0.replay() }.map_err(|err| ExecutionError::WithContext { reason: format!("{err}") })
    }
}

impl<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Clone
    for DeviceBackend<R, F, I, BT>
{
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Default
    for DeviceBackend<R, F, I, BT>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "fusion"))]
impl<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendIr
    for DeviceBackend<R, F, I, BT>
{
    type Handle = RudaTensor<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor
    }
}
