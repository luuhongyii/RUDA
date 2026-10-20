use std::marker::PhantomData;

use ruda_tensor::Shape;
use ruda_communication::ProtocolClient;
use ruda_tensor::graph::TensorIr;
use ruda_tensor_router::{RouterTensor, RunnerChannel, RunnerClient, get_client};

use super::{
    RemoteClient,
    runner::{RemoteBridge, RemoteDevice, RemoteTensorHandle},
};

/// A local channel with direct connection to the backend runner clients.
pub struct RemoteChannel<C: ProtocolClient> {
    _p: PhantomData<C>,
}

#[cfg(feature = "distributed")]
pub type ActiveChannel<C> = ruda_tensor_router::NativeDistributedChannel<RemoteChannel<C>>;
#[cfg(not(feature = "distributed"))]
pub type ActiveChannel<C> = RemoteChannel<C>;

impl<C: ProtocolClient> RunnerChannel for RemoteChannel<C> {
    type Device = RemoteDevice;
    type Bridge = RemoteBridge<C>;
    type Client = RemoteClient;

    type FloatElem = f32;

    type IntElem = i32;

    type BoolElem = u32;

    fn name(device: &Self::Device) -> String {
        format!("remote-{device:?}")
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        RemoteClient::init::<C>(device.clone())
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> RemoteTensorHandle<C> {
        RemoteTensorHandle {
            client: client.clone(),
            tensor: tensor.clone(),
            _p: PhantomData,
        }
    }

    fn register_tensor(
        client: &Self::Client,
        handle: RemoteTensorHandle<C>,
        shape: Shape,
        dtype: ruda_tensor::DType,
    ) -> RouterTensor<Self::Client> {
        assert_eq!(client.context_id(), handle.client.context_id(), "Transferred handle belongs to another remote session");
        assert_eq!(shape, handle.tensor.shape, "Transferred tensor shape changed");
        assert_eq!(dtype, handle.tensor.dtype, "Transferred tensor dtype changed");
        RouterTensor::new(handle.tensor.id, shape, dtype, client.clone())
    }

    fn change_client_backend(
        tensor: RouterTensor<Self::Client>,
        target_device: &Self::Device, // target device
    ) -> RouterTensor<Self::Client> {
        // Get tensor handle from current client
        let original_client = tensor.client.clone();
        let desc = tensor.into_ir();
        let handle = Self::get_tensor_handle(&desc, &original_client);

        let handle = handle.change_backend(target_device);

        let id = handle.tensor.id;

        let target_client = get_client::<ActiveChannel<C>>(target_device);
        let router_tensor: RouterTensor<RemoteClient> =
            RouterTensor::new(id, handle.tensor.shape, handle.tensor.dtype, target_client);

        router_tensor
    }
}

impl<C: ProtocolClient> Clone for RemoteChannel<C> {
    fn clone(&self) -> Self {
        RemoteChannel { _p: PhantomData }
    }
}

#[cfg(feature = "distributed")]
impl<C: ProtocolClient> ruda_tensor_router::NativeCollectiveChannel for RemoteChannel<C> {
    fn init_native_client(device: &RemoteDevice) -> RemoteClient { Self::init_client(device) }

    fn sync_native(client: &RemoteClient) {
        client.sync_native_collective().unwrap_or_else(|error| panic!("Remote collective synchronization failed: {error:?}"));
    }

    fn native_group(device: &RemoteDevice, devices: &[ruda_tensor::DeviceId]) -> Vec<ruda_tensor::DeviceId> {
        use ruda_core::device::Device;
        let info = get_client::<ActiveChannel<C>>(device).device_info()
            .unwrap_or_else(|error| panic!("Remote collective device discovery failed: {error:?}"));
        assert!(info.native_collectives, "The selected remote runner has no native collective capability");
        devices.iter().map(|id| {
            let participant = RemoteDevice::from_id(*id);
            assert!(participant.address.same_peer(&device.address), "Native remote collectives must use one server");
            assert!(participant.device_index < info.device_count, "Remote collective participant is not hosted");
            ruda_tensor::DeviceId::new(0, u16::try_from(participant.device_index).expect("Remote collective index exceeds u16"))
        }).collect()
    }
}
