use super::{ConnectError, RemoteDevice};
use ruda_communication::Protocol;

/// An endpoint exposing an ordered list of compute devices.
#[derive(Clone, Debug)]
pub struct RemoteHost {
    address: ruda_communication::Address,
    credential: crate::Credential,
}

impl RemoteHost {
    #[cfg(not(target_family = "wasm"))]
    pub fn websocket(address: impl Into<String>) -> Self {
        Self { address: address.into().parse().expect("Invalid remote address"), credential: crate::Credential::default() }
    }

    #[cfg(feature = "iroh")]
    pub fn iroh(host: impl Into<crate::IrohHost>) -> Self {
        Self { address: host.into().into(), credential: crate::Credential::default() }
    }

    pub fn with_credential(mut self, credential: impl Into<crate::Credential>) -> Self {
        self.credential = credential.into();
        self
    }

    /// Connect to the selected device and receive its server metadata.
    pub async fn connect_async(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        self.address.validate().map_err(|reason| ConnectError::InvalidConfiguration { reason })?;
        let device = RemoteDevice::with_address(self.address.clone(), index, self.credential.clone());
        let client = crate::remote_client(&device);
        if let Err(error) = client.connected().await {
            ruda_tensor_router::unregister_client::<super::ActiveChannel<<crate::shared::RemoteProtocol as Protocol>::Client>>(&device, &client);
            return Err(error);
        }
        Ok(device)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn connect(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        ruda_core::future::block_on(self.connect_async(index))
    }

    /// Discover the hosted devices; devices beyond index zero connect on first use.
    pub async fn devices_async(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        let first = self.connect_async(0).await?;
        let info = crate::remote_client(&first).connected().await?;
        Ok(core::iter::once(first).chain((1..info.device_count)
            .map(|index| RemoteDevice::with_address(self.address.clone(), index as usize, self.credential.clone()))).collect())
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn devices(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        ruda_core::future::block_on(self.devices_async())
    }
}
