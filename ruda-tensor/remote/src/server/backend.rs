use ruda_tensor::graph::{BackendIr, CustomOpIr, HandleContainer};
use ruda_tensor_router::{CustomOpRegistry, Runner};
use tokio_util::sync::CancellationToken;
use super::{PeerAuthorizer, ServerSettings, ServeError, Transport};

/// A configured remote server for a backend, with hosted device positions preserved.
pub struct BackendServer<B: BackendIr> {
    devices: Vec<B::Device>,
    settings: ServerSettings,
    custom_ops: CustomOpRegistry<B>,
    configure_runner: Option<fn(Runner<B>) -> Runner<B>>,
}

impl<B: BackendIr> BackendServer<B> {
    pub fn new(devices: impl IntoIterator<Item = B::Device>) -> Self {
        Self { devices: devices.into_iter().collect(), settings: ServerSettings::default(),
            custom_ops: CustomOpRegistry::new(), configure_runner: None }
    }

    pub fn with_settings(mut self, settings: ServerSettings) -> Self { self.settings = settings; self }

    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.settings = self.settings.with_authorizer(authorizer);
        self
    }

    pub fn with_telemetry(mut self, probe: crate::telemetry::TelemetryProbe) -> Self {
        self.settings = self.settings.with_telemetry(probe);
        self
    }

    pub fn with_custom_op<F>(mut self, id: &str, handler: F) -> Self
    where F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static {
        self.custom_ops.register(id, handler);
        self
    }

    pub fn with_custom_ops(mut self, registry: CustomOpRegistry<B>) -> Self {
        self.custom_ops = registry;
        self
    }

    #[cfg(feature = "distributed")]
    pub fn with_native_collectives(mut self) -> Self
    where B: ruda_tensor::distributed::DistributedBackend {
        self.configure_runner = Some(Runner::with_native_collectives);
        self
    }

    fn runners(&self) -> Result<Vec<Runner<B>>, ServeError> {
        if self.devices.is_empty() { return Err(ServeError::NoDevices); }
        if self.devices.len() > u16::MAX as usize + 1 { return Err(ServeError::TooManyDevices); }
        Ok(self.devices.iter().cloned().map(|device| {
            let runner = Runner::with_custom_ops(device, self.custom_ops.clone());
            match self.configure_runner { Some(configure) => configure(runner), None => runner }
        }).collect())
    }

    /// Block until process shutdown, including when called from inside another async runtime.
    pub fn serve(&self, transport: impl Into<Transport>) -> Result<(), ServeError> {
        let runners = self.runners()?;
        let settings = self.settings.clone();
        let transport = transport.into();
        std::thread::Builder::new().name("ruda-remote-server".into()).spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()
                .map_err(|error| ServeError::Runtime(error.to_string()))?;
            runtime.block_on(async move {
                let listener = transport.bind().await?;
                let stop = CancellationToken::new();
                let _stop_on_drop = stop.clone().drop_guard();
                let token = stop.clone();
                let serving = listener.serve(runners, settings, async move { token.cancelled().await; });
                tokio::pin!(serving);
                tokio::select! {
                    result = &mut serving => result,
                    signal = ruda_communication::util::try_os_shutdown_signal() => {
                        stop.cancel();
                        let result = serving.await;
                        signal.map_err(ServeError::SignalHandler)?;
                        result
                    }
                }
            })
        }).map_err(|error| ServeError::Runtime(error.to_string()))?
            .join().map_err(|_| ServeError::Runtime("server thread panicked".into()))?
    }

    /// Serve on the caller's runtime; dropping the returned future closes its sessions.
    pub fn serve_async<T: Into<Transport>>(&self, transport: T)
        -> impl Future<Output = Result<(), ServeError>> + Send + 'static + use<B, T> {
        self.serve_until(transport, core::future::pending())
    }

    /// Serve until the supplied future resolves, then drain sessions and close data links.
    pub fn serve_until<T: Into<Transport>, F: Future<Output = ()> + Send + 'static>(&self, transport: T, shutdown: F)
        -> impl Future<Output = Result<(), ServeError>> + Send + 'static + use<B, T, F> {
        let runners = self.runners();
        let settings = self.settings.clone();
        let transport = transport.into();
        async move {
            let runners = runners?;
            transport.bind().await?.serve(runners, settings, shutdown).await
        }
    }

    /// Attach these devices to an application-owned Iroh endpoint.
    #[cfg(feature = "iroh")]
    pub fn into_protocol(self, endpoint: &crate::Endpoint) -> Result<super::IrohRemoteProtocol<B>, ServeError> {
        let runners = self.runners()?;
        let node = crate::IrohNode::for_endpoint(endpoint.clone()).map_err(|error| ServeError::Bind(error.to_string()))?;
        let server = ruda_communication::iroh::IrohServer::new(node);
        super::RemoteServer::<B, ruda_communication::iroh::Iroh>::into_protocol(runners, server, self.settings)
    }
}
