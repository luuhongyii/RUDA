use std::{hash::{Hash, Hasher}, str::FromStr};
#[cfg(not(target_family = "wasm"))]
use std::net::SocketAddr;
use ::iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl,
    endpoint::{Builder, presets}};
#[cfg(not(target_family = "wasm"))]
use ::iroh::endpoint::QuicTransportConfig;
use serde::{Deserialize, Serialize};
use super::{IrohError, IrohNode};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IrohRelays {
    #[default]
    Public,
    Private { url: RelayUrl },
    Disabled,
}

impl FromStr for IrohRelays {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "public" => Ok(Self::Public),
            "disabled" => Ok(Self::Disabled),
            url => url.parse::<RelayUrl>().map(|url| Self::Private { url }).map_err(|error| error.to_string()),
        }
    }
}

impl core::fmt::Display for IrohRelays {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Public => f.write_str("public"),
            Self::Disabled => f.write_str("disabled"),
            Self::Private { url } => core::fmt::Display::fmt(url, f),
        }
    }
}

impl IrohRelays {
    pub(crate) fn builder(&self, segmentation_offload: bool) -> Builder {
        let builder = match self {
            Self::Public => Endpoint::builder(presets::N0),
            Self::Private { url } => Endpoint::builder(presets::Minimal).relay_mode(RelayMode::custom([url.clone()])),
            Self::Disabled => Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled),
        };
        #[cfg(not(target_family = "wasm"))]
        let builder = builder.transport_config(QuicTransportConfig::builder().enable_segmentation_offload(segmentation_offload).build());
        #[cfg(target_family = "wasm")]
        let _ = segmentation_offload;
        builder
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IrohHost {
    addr: EndpointAddr,
    relays: Option<IrohRelays>,
    #[serde(default)]
    segmentation_offload: Option<bool>,
    #[serde(skip)]
    endpoint: Option<Endpoint>,
}

impl IrohHost {
    pub fn new(server: impl Into<EndpointAddr>) -> Self { Self { addr: server.into(), relays: None, segmentation_offload: None, endpoint: None } }
    pub fn id(&self) -> EndpointId { self.addr.id }
    pub fn application_endpoint_id(&self) -> Option<EndpointId> { self.endpoint.as_ref().map(Endpoint::id) }
    pub fn with_relays(mut self, relays: IrohRelays) -> Self { self.relays = Some(relays); self }
    pub fn with_segmentation_offload(mut self, enabled: bool) -> Self { self.segmentation_offload = Some(enabled); self }
    #[cfg(not(target_family = "wasm"))]
    pub fn with_address(mut self, address: SocketAddr) -> Self { self.addr = self.addr.with_ip_addr(address); self }
    pub fn with_endpoint(mut self, endpoint: Endpoint) -> Self { self.endpoint = Some(endpoint); self }

    pub fn validate(&self) -> Result<(), String> {
        if self.endpoint.is_some() && (self.relays.is_some() || self.segmentation_offload.is_some()) {
            return Err("An application endpoint already owns its relay and segmentation-offload settings".into());
        }
        #[cfg(target_family = "wasm")]
        if self.relays == Some(IrohRelays::Disabled) { return Err("A browser Iroh endpoint requires a relay".into()); }
        #[cfg(not(target_family = "wasm"))]
        if self.endpoint.is_none() && self.relays == Some(IrohRelays::Disabled) && self.addr.ip_addrs().next().is_none() {
            return Err("An Iroh host with disabled relays needs a direct address".into());
        }
        Ok(())
    }

    pub fn dial_addr(&self) -> EndpointAddr {
        match &self.relays {
            Some(IrohRelays::Private { url }) => self.addr.clone().with_relay_url(url.clone()),
            _ => self.addr.clone(),
        }
    }

    pub(crate) async fn node(&self) -> Result<IrohNode, IrohError> {
        self.validate().map_err(IrohError)?;
        match &self.endpoint {
            Some(endpoint) => IrohNode::for_endpoint(endpoint.clone()),
            None => IrohNode::for_relays(self.relays.clone().unwrap_or_default(), self.segmentation_offload
                .unwrap_or_else(|| ruda_tensor_config::config().remote.iroh_segmentation_offload)).await,
        }
    }
}

impl PartialEq for IrohHost {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr && self.relays == other.relays && self.segmentation_offload == other.segmentation_offload
            && self.endpoint.as_ref().map(Endpoint::id) == other.endpoint.as_ref().map(Endpoint::id)
    }
}
impl Eq for IrohHost {}
impl Hash for IrohHost {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr.hash(state);
        self.relays.hash(state);
        self.segmentation_offload.hash(state);
        self.endpoint.as_ref().map(Endpoint::id).hash(state);
    }
}
impl From<EndpointId> for IrohHost { fn from(id: EndpointId) -> Self { Self::new(id) } }
impl From<EndpointAddr> for IrohHost { fn from(addr: EndpointAddr) -> Self { Self::new(addr) } }
