use super::*;
use crate::execution::capture::{self, CaptureCommunication, CaptureCommunicationStream, checked};
use cudarc::driver::sys::*;

impl CudaServer {
    pub(super) fn collective_stream(&mut self, stream_id: StreamId, compute: CUstream) -> Result<CUstream, ServerError> {
        let recording = self.capture_group(stream_id);
        let root = recording.or_else(|| self.capture_warmup_root(stream_id));
        let communication = if let Some(root) = root {
            if !self.captures.communication.contains_key(&root) {
                let stream = CaptureCommunicationStream::new(self.ctx.context as usize)?;
                capture::dependency(self.comm_stream, stream.raw as CUstream)?;
                self.captures.communication.insert(root, CaptureCommunication { stream, joined: false });
            }
            let state = self.captures.communication.get(&root).expect("capture communication stream");
            let communication = state.stream.raw as CUstream;
            if recording.is_some() && !state.joined {
                checked("cuStreamSynchronize before collective capture", unsafe { cuStreamSynchronize(communication) })?;
            }
            communication
        } else { self.comm_stream };
        capture::dependency(compute, communication)?;
        if let Some(root) = recording {
            self.captures.communication.get_mut(&root).expect("capture communication stream").joined = true;
        }
        Ok(communication)
    }

    pub(super) fn collective_sync_stream(&mut self, stream_id: StreamId, compute: CUstream) -> Result<(), ServerError> {
        let recording = self.capture_group(stream_id);
        let root = recording.or_else(|| self.capture_warmup_root(stream_id));
        let state = root.and_then(|root| self.captures.communication.get(&root));
        let communication = state.map_or(self.comm_stream, |state| state.stream.raw as CUstream);
        if recording.is_some() && !state.is_some_and(|state| state.joined) {
            checked("cuStreamSynchronize non-captured collectives", unsafe { cuStreamSynchronize(communication) })
        } else {
            capture::dependency(communication, compute)
        }
    }

    pub(super) fn collective_result<T>(&mut self, stream_id: StreamId, writes: &[Binding], result: Result<T, ServerError>) -> Result<T, ServerError> {
        if let Err(error) = &result {
            Binding::fail_all(writes.iter(), error.clone());
            if self.capture_owner(stream_id) == Some(stream_id) { self.capture_failure(stream_id, error); }
        }
        result
    }
}
