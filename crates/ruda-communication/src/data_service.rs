//! This module enables direct data transfer between servers without blocking the client or any server.
//!
//! It eliminates the need for intermediate data transfer through the client, avoiding the process of downloading data from one server and reuploading it to another.
//!
//! The module provides an optimized mechanism for servers to communicate directly, streamlining data movement between them without involving the client.

use crate::Message;
use crate::base::Protocol;
use crate::base::{Address, CommunicationChannel, ProtocolClient, ProtocolServer};
use ruda_tensor::api::{TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

pub use crate::transfer::{TensorTransferId, TransferCapability};

const TRANSFER_LIFETIME: std::time::Duration = std::time::Duration::from_secs(300);

struct CapabilityState {
    bytes: bytes::Bytes,
    target: crate::PeerId,
    remaining: u32,
    expires: tokio::time::Instant,
    owner: CancellationToken,
    retired: CancellationToken,
    generation: Arc<()>,
}
impl Drop for CapabilityState { fn drop(&mut self) { self.retired.cancel(); } }

#[derive(Debug, Serialize, Deserialize)]
enum DataServiceMessage {
    TensorRequest(TensorTransferId),
    Tensor(TensorData),
    CapabilityRequest(TransferCapability),
    Denied(String),
}

type ClientChannelRef<C> = Arc<Mutex<<C as ProtocolClient>::Channel>>;

pub struct TensorDataService<B: Backend, P: Protocol<Client: ProtocolClient>> {
    /// Maps tensor transfer IDs to their exposed state.
    pub exposed_tensors: Mutex<HashMap<TensorTransferId, TensorExposeState>>,
    /// Maps node addresses to their channels.
    pub channels: Mutex<HashMap<Address, ClientChannelRef<P::Client>>>,
    /// Notify when a new tensor is exposed.
    pub new_tensor_notify: Arc<Notify>,

    cancel_token: CancellationToken,
    connector: crate::ClientConnector<P::Client>,
    capabilities: Arc<Mutex<HashMap<TransferCapability, CapabilityState>>>,

    _phantom_data: PhantomData<B>,
}

pub struct TensorExposeState {
    /// The bytes of the tensor data message. Message::Data(...) serialized with rmp_serde
    pub bytes: bytes::Bytes,
    /// How many times the tensor will be downloaded
    pub max_downloads: u32,
    /// How man times the tensor has been downloaded
    pub cur_download_count: u32,
}

/// Provides a routing function for a tensor data service for a communications server
pub trait TensorDataServer<B: Backend, P: Protocol> {
    /// Routes the tensor data service to the "/data" route
    fn route_tensor_data_service(self, state: Arc<TensorDataService<B, P>>) -> Self;
}

impl<B: Backend, S: ProtocolServer + Sized, P: Protocol<Server = S> + 'static>
    TensorDataServer<B, P> for S
{
    fn route_tensor_data_service(self, state: Arc<TensorDataService<B, P>>) -> Self {
        self.route("/data", async move |stream: S::Channel| {
            state.handle_data_channel(stream).await;
        })
    }
}

impl<B: Backend, P: Protocol> TensorDataService<B, P> {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            exposed_tensors: Mutex::new(HashMap::new()),
            channels: Mutex::new(HashMap::new()),
            new_tensor_notify: Arc::new(Notify::new()),
            cancel_token,
            connector: Arc::new(|address, route| P::Client::connect(address, &route)),
            capabilities: Arc::new(Mutex::new(HashMap::new())),
            _phantom_data: PhantomData::<B>,
        }
    }

    pub fn with_connector(mut self, connector: crate::ClientConnector<P::Client>) -> Self {
        self.connector = connector;
        self
    }

    /// Exposes a tensor to the data server, allowing it to be downloaded by other nodes.
    pub async fn expose(
        &self,
        tensor: B::FloatTensorPrimitive,
        max_downloads: u32,
        transfer_id: TensorTransferId,
    ) {
        let data = B::float_into_data(tensor).await.unwrap();
        self.expose_data(data, max_downloads, transfer_id).await
    }

    /// Exposes a tensor data to the data server, allowing it to be downloaded by other nodes.
    pub async fn expose_data(
        &self,
        tensor_data: TensorData,
        max_downloads: u32,
        transfer_id: TensorTransferId,
    ) {
        let bytes: bytes::Bytes = rmp_serde::to_vec(&DataServiceMessage::Tensor(tensor_data))
            .unwrap()
            .into();
        let mut exposed_tensors = self.exposed_tensors.lock().await;
        exposed_tensors.insert(
            transfer_id,
            TensorExposeState {
                bytes,
                max_downloads,
                cur_download_count: 0,
            },
        );
        core::mem::drop(exposed_tensors);
        self.new_tensor_notify.notify_waiters();
    }

    pub async fn close(&self) {
        let streams: Vec<_> = self.channels.lock().await.drain().map(|(_, stream)| stream).collect();
        for stream in streams {
            if let Err(error) = stream.lock().await.close().await {
                log::warn!("Failed to close data channel: {error:?}");
            }
        }
    }

    /// Stop the data service, releasing unused data and interrupting its waiters.
    pub async fn shutdown(&self) {
        self.cancel_token.cancel();
        self.exposed_tensors.lock().await.clear();
        self.capabilities.lock().await.clear();
        self.close().await;
    }

    pub async fn download_tensor(&self, remote: Address, transfer_id: TensorTransferId) -> Option<TensorData> {
        self.download_tensor_cancelable(remote, transfer_id, &self.cancel_token).await
    }

    /// Cancel one download without allowing its outstanding reply to become another request's data.
    pub async fn download_tensor_cancelable(&self, remote: Address, transfer_id: TensorTransferId,
        cancelled: &CancellationToken) -> Option<TensorData> {
        loop {
            let stream = tokio::select! {
                _ = self.cancel_token.cancelled() => return None,
                _ = cancelled.cancelled() => return None,
                stream = self.get_data_stream(remote.clone()) => stream?,
            };
            let mut channel = tokio::select! {
                _ = self.cancel_token.cancelled() => return None,
                _ = cancelled.cancelled() => return None,
                channel = stream.lock() => channel,
            };
            let current = self.channels.lock().await.get(&remote)
                .is_some_and(|current| Arc::ptr_eq(current, &stream));
            // Another request may have invalidated this connection while this one waited.
            if !current { continue; }
            let download = async {
                let bytes = rmp_serde::to_vec(&DataServiceMessage::TensorRequest(transfer_id)).ok()?;
                channel.send(Message::new(bytes.into())).await.ok()?;
                let message = channel.recv().await.ok()??;
                match rmp_serde::from_slice::<DataServiceMessage>(&message.data).ok()? {
                    DataServiceMessage::Tensor(data) => Some(data),
                    _ => None,
                }
            };
            let result = tokio::select! {
                _ = self.cancel_token.cancelled() => None,
                _ = cancelled.cancelled() => None,
                result = download => result,
            };
            if result.is_none() {
                let mut channels = self.channels.lock().await;
                if channels.get(&remote).is_some_and(|current| Arc::ptr_eq(current, &stream)) {
                    channels.remove(&remote);
                }
            }
            return result;
        }
    }

    async fn get_data_stream(&self, address: Address) -> Option<ClientChannelRef<P::Client>> {
        if let Some(stream) = self.channels.lock().await.get(&address).cloned() { return Some(stream); }
        let stream = Arc::new(Mutex::new((self.connector)(address.clone(), "data".into()).await?));
        let mut streams = self.channels.lock().await;
        Some(streams.entry(address).or_insert(stream).clone())
    }

    /// Get the requested exposed tensor data, and update download counter
    async fn get_exposed_tensor_bytes(
        &self,
        transfer_id: TensorTransferId,
    ) -> Option<bytes::Bytes> {
        loop {
            let notified = self.new_tensor_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut exposed_tensors = self.exposed_tensors.lock().await;
                // take the tensor out of the hashmap while we download
                if let Some(mut exposed_state) = exposed_tensors.remove(&transfer_id) {
                    exposed_state.cur_download_count += 1;
                    let bytes = if exposed_state.cur_download_count == exposed_state.max_downloads {
                        exposed_state.bytes
                    } else {
                        let bytes = exposed_state.bytes.clone();
                        exposed_tensors.insert(transfer_id, exposed_state);
                        bytes
                    };
                    return Some(bytes);
                }
            }
            // No matching tensor, wait for a new one to come in.
            tokio::select! {
                _ = notified => {},
                _ = self.cancel_token.cancelled() => return None,
            }
        }
    }

    /// Handle incoming connections for downloading tensors.
    pub(crate) async fn handle_data_channel(&self, mut channel: <P::Server as ProtocolServer>::Channel) {
        loop {
            let message = tokio::select! {
                _ = self.cancel_token.cancelled() => return,
                message = channel.recv() => match message { Ok(Some(message)) => message, _ => return },
            };
            let transfer_id = match rmp_serde::from_slice::<DataServiceMessage>(&message.data) {
                Ok(DataServiceMessage::CapabilityRequest(capability)) => {
                    self.handle_capability_channel(channel, capability).await;
                    return;
                }
                Ok(DataServiceMessage::TensorRequest(id)) if channel.authenticated_peer().is_none() => id,
                _ => { log::warn!("Invalid data-service request"); return; }
            };
            let bytes = tokio::select! {
                _ = self.cancel_token.cancelled() => return,
                _ = channel.recv() => return,
                bytes = self.get_exposed_tensor_bytes(transfer_id) => match bytes { Some(bytes) => bytes, None => return },
            };
            tokio::select! {
                _ = self.cancel_token.cancelled() => return,
                result = channel.send(Message::new(bytes)) => if result.is_err() { return; },
            }
        }
    }
}

impl<B: Backend, P: Protocol> TensorDataService<B, P> {
    pub async fn expose_capability(&self, data: TensorData, count: u32, capability: TransferCapability,
        target: crate::PeerId, owner: CancellationToken) -> Result<(), String> {
        if !matches!(target, crate::PeerId::Iroh(_)) { return Err("An authenticated tensor transfer needs an Iroh target".into()); }
        if count == 0 { return Err("A tensor capability needs at least one download".into()); }
        if owner.is_cancelled() || self.cancel_token.is_cancelled() { return Err("Tensor exposure session is closed".into()); }
        let bytes = rmp_serde::to_vec(&DataServiceMessage::Tensor(data)).map_err(|error| error.to_string())?.into();
        let generation = Arc::new(());
        let retired = CancellationToken::new();
        let expires = tokio::time::Instant::now() + TRANSFER_LIFETIME;
        {
            let mut capabilities = self.capabilities.lock().await;
            if capabilities.contains_key(&capability) { return Err("Tensor capability is already exposed".into()); }
            capabilities.insert(capability, CapabilityState {
                bytes, target, remaining: count, expires, owner: owner.clone(), retired: retired.clone(), generation: generation.clone(),
            });
        }
        self.new_tensor_notify.notify_waiters();
        let capabilities = self.capabilities.clone();
        let notify = self.new_tensor_notify.clone();
        let shutdown = self.cancel_token.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep_until(expires) => {},
                _ = owner.cancelled() => {},
                _ = retired.cancelled() => {},
                _ = shutdown.cancelled() => {},
            }
            let mut capabilities = capabilities.lock().await;
            if capabilities.get(&capability).is_some_and(|entry| Arc::ptr_eq(&entry.generation, &generation)) {
                capabilities.remove(&capability);
            }
            drop(capabilities);
            notify.notify_waiters();
        });
        Ok(())
    }

    async fn take_capability(&self, capability: TransferCapability, peer: &crate::PeerId) -> Result<bytes::Bytes, String> {
        tokio::time::timeout(TRANSFER_LIFETIME, async {
            loop {
                let notified = self.new_tensor_notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                {
                    let mut capabilities = self.capabilities.lock().await;
                    if let Some(entry) = capabilities.get_mut(&capability) {
                        if &entry.target != peer { return Err("Tensor capability belongs to another peer".into()); }
                        if entry.owner.is_cancelled() || entry.expires <= tokio::time::Instant::now() {
                            capabilities.remove(&capability);
                            return Err("Tensor capability expired or its session closed".into());
                        }
                        let bytes = entry.bytes.clone();
                        entry.remaining -= 1;
                        if entry.remaining == 0 { capabilities.remove(&capability); }
                        return Ok(bytes);
                    }
                }
                tokio::select! {
                    _ = self.cancel_token.cancelled() => return Err("Tensor data service closed".into()),
                    _ = notified => {},
                }
            }
        }).await.map_err(|_| "Timed out waiting for a tensor capability".to_string())?
    }

    async fn handle_capability_channel(&self, mut channel: <P::Server as ProtocolServer>::Channel, capability: TransferCapability) {
        let result = match channel.authenticated_peer() {
            Some(peer) => tokio::select! {
                _ = self.cancel_token.cancelled() => return,
                _ = channel.recv() => return,
                result = self.take_capability(capability, &peer) => result,
            },
            None => Err("Tensor capability requests require an authenticated transport".into()),
        };
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(reason) => match rmp_serde::to_vec(&DataServiceMessage::Denied(reason)) { Ok(bytes) => bytes.into(), Err(_) => return },
        };
        tokio::select! {
            _ = self.cancel_token.cancelled() => {},
            _ = async { let _ = channel.send(Message::new(bytes)).await; let _ = channel.close().await; } => {},
        }
    }

    pub async fn download_capability(&self, remote: Address, capability: TransferCapability,
        cancelled: &CancellationToken) -> Option<TensorData> {
        let download = async {
            let peer = remote.peer_id();
            let mut channel = (self.connector)(remote, "data".into()).await?;
            if channel.authenticated_peer().as_ref() != Some(&peer) { return None; }
            let bytes = rmp_serde::to_vec(&DataServiceMessage::CapabilityRequest(capability)).ok()?;
            channel.send(Message::new(bytes.into())).await.ok()?;
            let message = channel.recv().await.ok()??;
            match rmp_serde::from_slice::<DataServiceMessage>(&message.data).ok()? {
                DataServiceMessage::Tensor(data) => Some(data),
                DataServiceMessage::Denied(reason) => { log::warn!("Tensor transfer refused: {reason}"); None }
                _ => None,
            }
        };
        tokio::select! {
            _ = self.cancel_token.cancelled() => None,
            _ = cancelled.cancelled() => None,
            result = download => result,
        }
    }
}
