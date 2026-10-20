use std::{fmt, net::SocketAddr, sync::Arc};
use crate::Credential;

/// Iroh public keys are transport-authenticated; WebSocket addresses are not identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClientId { WebSocket(SocketAddr), Iroh([u8; 32]), Unknown }

#[derive(Debug)]
pub struct AuthorizationRequest<'a> {
    pub client: ClientId,
    pub device_index: u32,
    pub credential: &'a Credential,
}

pub trait PeerAuthorizer: Send + Sync + 'static {
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String>;
}
impl<F> PeerAuthorizer for F where F: Fn(AuthorizationRequest<'_>) -> Result<(), String> + Send + Sync + 'static {
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String> { self(request) }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllowAll;
impl PeerAuthorizer for AllowAll {
    fn authorize(&self, _: AuthorizationRequest<'_>) -> Result<(), String> { Ok(()) }
}

/// Optional shared-token policy; WebSocket itself does not encrypt the token.
#[derive(Clone)]
pub struct TokenAuthorizer { digest: blake3::Hash }
impl TokenAuthorizer {
    pub fn new(token: impl Into<Credential>) -> Result<Self, EmptyToken> {
        let token = token.into();
        if token.is_empty() { return Err(EmptyToken); }
        Ok(Self { digest: blake3::hash(token.as_bytes()) })
    }
}
impl fmt::Debug for TokenAuthorizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("TokenAuthorizer").finish_non_exhaustive() }
}
impl PeerAuthorizer for TokenAuthorizer {
    fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<(), String> {
        if blake3::hash(request.credential.as_bytes()) == self.digest { Ok(()) }
        else { Err(format!("Client {:?} presented the wrong token", request.client)) }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptyToken;
impl fmt::Display for EmptyToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("A token authorizer needs a nonempty token") }
}
impl std::error::Error for EmptyToken {}

#[derive(Clone)]
pub struct ServerSettings {
    pub(crate) authorizer: Arc<dyn PeerAuthorizer>,
    pub(crate) telemetry: crate::telemetry::TelemetryProbe,
}
impl Default for ServerSettings {
    fn default() -> Self { Self { authorizer: Arc::new(AllowAll), telemetry: crate::telemetry::TelemetryProbe::new(crate::telemetry::CHANNEL_CAPACITY) } }
}
impl ServerSettings {
    pub fn with_telemetry(mut self, probe: crate::telemetry::TelemetryProbe) -> Self { self.telemetry = probe; self }
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.authorizer = Arc::new(authorizer);
        self
    }
}
