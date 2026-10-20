use ruda_tensor::{DeviceOps, graph::{BackendIr, CustomOpIr, HandleContainer}};
use crate::{Dispatch, DispatchDevice};

pub use ruda_tensor_remote::server::{BackendServer, CustomOpRegistry, PeerAuthorizer, ServerSettings, ServeError, Transport, WebSocketTransport};
#[cfg(feature = "iroh")]
pub use ruda_tensor_remote::{Endpoint, IrohIdentity, IrohRelays, IrohTransport, RUDA_REMOTE_ALPN};
#[cfg(feature = "iroh")]
pub type RemoteProtocol = ruda_tensor_remote::server::IrohRemoteProtocol<Dispatch>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family { Host, Device }

#[derive(Clone, Copy)]
enum DeviceSetError { MixedBackends, Unsupported(&'static str) }
impl DeviceSetError {
    fn into_error(self) -> ServeError {
        match self {
            Self::MixedBackends => ServeError::MixedBackends,
            Self::Unsupported(reason) => ServeError::UnsupportedDevice { reason: reason.into() },
        }
    }
}

fn family(device: &DispatchDevice) -> Result<Family, DeviceSetError> {
    match device.inner() {
        DispatchDevice::Host(_) => Ok(Family::Host),
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(_) => Ok(Family::Device),
        #[cfg(feature = "cpu")]
        DispatchDevice::Cpu(_) => Ok(Family::Device),
        #[cfg(feature = "wgpu")]
        DispatchDevice::Wgpu(_) => Ok(Family::Device),
        #[cfg(feature = "rocm")]
        DispatchDevice::Rocm(_) => Ok(Family::Device),
        #[cfg(feature = "tch")]
        DispatchDevice::LibTorch(_) => Err(DeviceSetError::Unsupported("LibTorch is not a hosted Dispatch backend")),
        DispatchDevice::Remote(_) => Err(DeviceSetError::Unsupported("a remote device cannot host a remote server")),
        #[cfg(feature = "capture")]
        DispatchDevice::Capture(_) => Err(DeviceSetError::Unsupported("a capture device cannot host a remote server")),
        DispatchDevice::Autodiff(..) => unreachable!("autodiff device already unwrapped"),
    }
}

/// Serve runtime-selected local devices; the autodiff graph stays on the client.
/// Device-runtime variants may share a server, but cannot be mixed with Host devices.
pub struct RemoteServer {
    backend: BackendServer<Dispatch>,
    device_error: Option<DeviceSetError>,
    #[cfg(feature = "distributed")]
    native_collectives: bool,
}

impl RemoteServer {
    pub fn new(devices: impl IntoIterator<Item = DispatchDevice>) -> Self {
        let devices: Vec<_> = devices.into_iter().map(|device| device.inner().clone()).collect();
        let families = devices.iter().map(family).collect::<Result<Vec<_>, _>>();
        let device_error = match families {
            Err(error) => Some(error),
            Ok(families) if families.windows(2).any(|pair| pair[0] != pair[1]) => Some(DeviceSetError::MixedBackends),
            Ok(_) => None,
        };
        #[cfg(feature = "distributed")]
        let native_collectives = !devices.is_empty() && devices.iter().all(|device| match device {
            #[cfg(feature = "cuda")]
            DispatchDevice::Cuda(_) => true,
            _ => false,
        });
        Self { backend: BackendServer::new(devices), device_error,
            #[cfg(feature = "distributed")]
            native_collectives }
    }

    fn check(&self) -> Result<(), ServeError> {
        self.device_error.map_or(Ok(()), |error| Err(error.into_error()))
    }

    pub fn with_settings(mut self, settings: ServerSettings) -> Self {
        self.backend = self.backend.with_settings(settings);
        self
    }

    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.backend = self.backend.with_authorizer(authorizer);
        self
    }

    pub fn with_telemetry(mut self, probe: ruda_tensor_remote::telemetry::TelemetryProbe) -> Self {
        self.backend = self.backend.with_telemetry(probe);
        self
    }

    pub fn with_custom_op<F>(mut self, id: &str, handler: F) -> Self
    where F: Fn(&mut HandleContainer<<Dispatch as BackendIr>::Handle>, &CustomOpIr, &DispatchDevice) + Send + Sync + 'static {
        self.backend = self.backend.with_custom_op(id, handler);
        self
    }

    pub fn with_custom_ops(mut self, registry: CustomOpRegistry<Dispatch>) -> Self {
        self.backend = self.backend.with_custom_ops(registry);
        self
    }

    #[cfg(feature = "distributed")]
    pub fn with_native_collectives(mut self) -> Result<Self, ServeError> {
        self.check()?;
        if !self.native_collectives {
            return Err(ServeError::UnsupportedDevice { reason: "native Dispatch server collectives require CUDA devices".into() });
        }
        self.backend = self.backend.with_native_collectives();
        Ok(self)
    }

    pub fn serve(&self, transport: impl Into<Transport>) -> Result<(), ServeError> {
        self.check()?;
        self.backend.serve(transport)
    }

    pub fn serve_async<T: Into<Transport>>(&self, transport: T)
        -> impl Future<Output = Result<(), ServeError>> + Send + 'static + use<T> {
        let checked = self.check();
        let serving = self.backend.serve_async(transport);
        async move { checked?; serving.await }
    }

    pub fn serve_until<T: Into<Transport>, F: Future<Output = ()> + Send + 'static>(&self, transport: T, shutdown: F)
        -> impl Future<Output = Result<(), ServeError>> + Send + 'static + use<T, F> {
        let checked = self.check();
        let serving = self.backend.serve_until(transport, shutdown);
        async move { checked?; serving.await }
    }

    #[cfg(feature = "iroh")]
    pub fn into_protocol(self, endpoint: &Endpoint) -> Result<RemoteProtocol, ServeError> {
        self.check()?;
        self.backend.into_protocol(endpoint)
    }
}
