use core::fmt;
use crate::shared::{PROTOCOL_VERSION, SessionRefusal};

/// Failure to establish or retain a remote compute session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectError {
    InvalidConfiguration { reason: String },
    Unreachable { reason: String },
    Handshake { reason: String },
    Unauthorized,
    NoSuchDevice { device_count: usize },
    IncompatibleProtocol { client_version: u16, server_version: u16 },
    InvalidSession,
    Interrupted,
    ConnectionClosed { reason: String },
}

impl From<SessionRefusal> for ConnectError {
    fn from(refusal: SessionRefusal) -> Self {
        match refusal {
            SessionRefusal::Unauthorized => Self::Unauthorized,
            SessionRefusal::NoSuchDevice { device_count } => Self::NoSuchDevice { device_count: device_count as usize },
            SessionRefusal::IncompatibleProtocol { server_version } if server_version == PROTOCOL_VERSION =>
                Self::Handshake { reason: "The server could not decode the session handshake".into() },
            SessionRefusal::IncompatibleProtocol { server_version } =>
                Self::IncompatibleProtocol { client_version: PROTOCOL_VERSION, server_version },
            SessionRefusal::InvalidSession => Self::InvalidSession,
        }
    }
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { reason } => write!(f, "Invalid remote configuration: {reason}"),
            Self::Unreachable { reason } => write!(f, "Remote server is unreachable: {reason}"),
            Self::Handshake { reason } => write!(f, "Remote handshake failed: {reason}"),
            Self::Unauthorized => f.write_str("Remote server refused the credential"),
            Self::NoSuchDevice { device_count } => write!(f, "Remote server hosts only {device_count} devices"),
            Self::IncompatibleProtocol { client_version, server_version } => write!(f, "Remote protocol mismatch: client {client_version}, server {server_version}"),
            Self::InvalidSession => f.write_str("Remote session is closed, already bound, or bound to another device"),
            Self::Interrupted => f.write_str("Remote runtime stopped"),
            Self::ConnectionClosed { reason } => write!(f, "Remote connection closed: {reason}"),
        }
    }
}
impl std::error::Error for ConnectError {}
