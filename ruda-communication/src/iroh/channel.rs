use ::iroh::endpoint::{Connection, SendStream, RecvStream};
use crate::ChannelFuture;
use crate::{Address, CommunicationChannel, CommunicationError, Message, PeerId, ProtocolClient};
use super::IrohNode;

const MAX_FRAME_SIZE: usize = 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct IrohError(pub String);
impl IrohError {
    pub(crate) fn from_error(error: impl core::fmt::Display) -> Self { Self(error.to_string()) }
}
impl core::fmt::Display for IrohError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for IrohError {}
impl CommunicationError for IrohError {}

pub struct IrohChannel {
    retained_node: Option<IrohNode>,
    connection: Connection,
    send: SendStream,
    recv: RecvStream,
    header: [u8; 8],
    header_read: usize,
    body: Option<Vec<u8>>,
    body_read: usize,
}

impl IrohChannel {
    pub(crate) fn new(connection: Connection, send: SendStream, recv: RecvStream) -> Self {
        Self { retained_node: None, connection, send, recv, header: [0; 8], header_read: 0, body: None, body_read: 0 }
    }

    pub(crate) fn retain_node(&mut self, node: IrohNode) { self.retained_node = Some(node); }
}

impl CommunicationChannel for IrohChannel {
    type Error = IrohError;
    fn authenticated_peer(&self) -> Option<PeerId> { Some(PeerId::Iroh(*self.connection.remote_id().as_bytes())) }

    async fn send(&mut self, message: Message) -> Result<(), IrohError> {
        if message.data.len() > MAX_FRAME_SIZE { return Err(IrohError("Iroh frame exceeds 1 GiB".into())); }
        self.send.write_all(&(message.data.len() as u64).to_le_bytes()).await.map_err(IrohError::from_error)?;
        self.send.write_all(&message.data).await.map_err(IrohError::from_error)
    }

    async fn recv(&mut self) -> Result<Option<Message>, IrohError> {
        while self.header_read < 8 {
            match self.recv.read(&mut self.header[self.header_read..]).await.map_err(IrohError::from_error)? {
                Some(read) => self.header_read += read,
                None if self.header_read == 0 => return Ok(None),
                None => return Err(IrohError("Truncated Iroh frame header".into())),
            }
        }
        if self.body.is_none() {
            let length = u64::from_le_bytes(self.header);
            if length > MAX_FRAME_SIZE as u64 { return Err(IrohError("Peer sent an oversized Iroh frame".into())); }
            self.body = Some(vec![0; length as usize]);
        }
        let body = self.body.as_mut().expect("frame header decoded");
        while self.body_read < body.len() {
            match self.recv.read(&mut body[self.body_read..]).await.map_err(IrohError::from_error)? {
                Some(read) => self.body_read += read,
                None => return Err(IrohError("Truncated Iroh frame body".into())),
            }
        }
        let data = self.body.take().expect("complete frame");
        self.header_read = 0;
        self.body_read = 0;
        Ok(Some(Message::new(data.into())))
    }

    async fn close(&mut self) -> Result<(), IrohError> {
        self.send.finish().map_err(IrohError::from_error)
    }
}

pub struct IrohClient;
impl IrohClient {
    pub async fn connect_from(node: &IrohNode, address: Address, route: &str) -> Result<IrohChannel, IrohError> {
        let host = address.iroh.ok_or_else(|| IrohError("An Iroh connection requires an Iroh address".into()))?;
        node.open(host.dial_addr(), route).await
    }
}
impl ProtocolClient for IrohClient {
    type Channel = IrohChannel;
    type Error = IrohError;
    fn connect(address: Address, route: &str) -> ChannelFuture<Option<IrohChannel>> {
        let route = route.to_owned();
        Box::pin(async move {
            let host = address.iroh?;
            let result = async { host.node().await?.open(host.dial_addr(), &route).await }.await;
            match result {
                Ok(channel) => Some(channel),
                Err(error) => { log::warn!("Iroh connection failed: {error}"); None }
            }
        })
    }
}
