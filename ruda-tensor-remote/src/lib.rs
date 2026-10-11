#[macro_use]
extern crate derive_new;

#[cfg(feature = "client")]
pub(crate) mod client;

#[cfg(feature = "server")]
pub mod server;

pub(crate) mod shared;
pub use ruda_tensor_router::{GraphBindings, GraphId, GraphIr, RunnerClient};

#[cfg(feature = "client")]
mod __client {
    use super::*;

    use crate::{client::RemoteChannel, shared::RemoteProtocol};
    use ruda_communication::Protocol;
    use ruda_tensor_router::BackendRouter;

    /// The remote backend allows you to run computation on a remote device.
    ///
    /// Make sure there is a running server before trying to connect to it.
    ///
    /// ```rust, ignore
    /// fn main() {
    ///     let device = Default::default();
    ///     let port = 3000;
    ///
    ///     // You need to activate the `server` feature flag to have access to this function.
    ///     ruda::server::start::<ruda::backend::Wgpu>(device, port);
    /// }
    ///```
    pub type RemoteBackend = BackendRouter<RemoteChannel<<RemoteProtocol as Protocol>::Client>>;

    pub use client::{RemoteClient, RemoteDevice};

    /// Obtain the same device client used by RemoteBackend for explicit graph operations.
    pub fn remote_client(device: &RemoteDevice) -> RemoteClient {
        ruda_tensor_router::get_client::<RemoteChannel<<RemoteProtocol as Protocol>::Client>>(device)
    }
}
#[cfg(feature = "client")]
pub use __client::*;

#[cfg(all(test, feature = "client", feature = "server"))]
mod tests {
    use crate::RemoteBackend;
    use ruda_tensor_host::Host;
    use ruda_tensor::api::{Distribution, Tensor};

    #[test]
    pub fn test_to_device_over_websocket() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(crate::server::start_websocket_async::<Host>(
            Default::default(),
            3000,
        ));
        rt.spawn(crate::server::start_websocket_async::<Host>(
            Default::default(),
            3010,
        ));

        let remote_device_1 = super::RemoteDevice::new("ws://127.0.0.1:3000");
        let remote_device_2 = super::RemoteDevice::new("ws://127.0.0.1:3010");

        // Some random input
        let input_shape = [1, 28, 28];
        let input = Tensor::<RemoteBackend, 3>::random(
            input_shape,
            Distribution::Default,
            &remote_device_1,
        );
        let numbers_expected: Vec<f32> = input.to_data().to_vec().unwrap();

        // Move tensor to device 2
        let input = input.to_device(&remote_device_2);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        // Move tensor back to device 1
        let input = input.to_device(&remote_device_1);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        rt.shutdown_background();
    }

    /// Two independent clients both number their tensors from zero; the server must keep them
    /// apart instead of letting the second registration replace the first.
    #[test]
    pub fn test_independent_clients_do_not_share_tensors() {
        use crate::{RemoteClient, RunnerClient, shared::RemoteProtocol};
        use ruda_communication::Protocol;
        use ruda_tensor::TensorData;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();
        rt.spawn(crate::server::start_websocket_async::<Host>(
            Default::default(),
            3020,
        ));
        std::thread::sleep(std::time::Duration::from_millis(500));

        type Client = <RemoteProtocol as Protocol>::Client;
        let device = super::RemoteDevice::new("ws://127.0.0.1:3020");
        let client_a = RemoteClient::init::<Client>(device.clone());
        let client_b = RemoteClient::init::<Client>(device);

        let a = client_a.register_tensor_data(TensorData::new(vec![1.0f32, 2.0], [2]));
        let b = client_b.register_tensor_data(TensorData::new(vec![9.0f32, 8.0], [2]));
        let a2 = client_a.register_tensor_data(TensorData::new(vec![3.0f32], [1]));

        let read = |client: &RemoteClient, tensor: ruda_tensor_router::RouterTensor<RemoteClient>| {
            let data = rt.block_on(client.read_tensor_async(tensor.into_ir())).unwrap();
            data.to_vec::<f32>().unwrap()
        };
        assert_eq!(read(&client_b, b), vec![9.0, 8.0]);
        assert_eq!(read(&client_a, a), vec![1.0, 2.0]);
        assert_eq!(read(&client_a, a2), vec![3.0]);

        rt.shutdown_background();
    }
}
