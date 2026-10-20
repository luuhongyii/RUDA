use super::*;
use crate::execution::{capture::{self, Captured, CaptureState, checked}, graph::error};
use cudarc::driver::sys::*;

impl CudaServer {
    pub(super) fn capture_owner(&self, stream_id: StreamId) -> Option<StreamId> {
        self.captures.active.iter().chain(&self.captures.preparing).chain(self.captures.members.keys())
            .chain(self.captures.warm_members.keys()).copied()
            .find(|owner| self.streams.shares_physical_stream(*owner, stream_id))
    }

    pub(super) fn capture_group(&self, stream_id: StreamId) -> Option<StreamId> {
        let owner = self.capture_owner(stream_id)?;
        if self.captures.active.contains(&owner) { Some(owner) }
        else { self.captures.members.get(&owner).copied() }
    }

    pub(super) fn capture_warmup_root(&self, stream_id: StreamId) -> Option<StreamId> {
        let owner = self.capture_owner(stream_id)?;
        let root = self.captures.warm_members.get(&owner).copied().unwrap_or(owner);
        self.captures.preparing.contains(&root).then_some(root)
    }

    pub(super) fn capture_prepare_branch(&mut self, stream_id: StreamId, root: StreamId) -> Result<(), ServerError> {
        if !self.captures.preparing.contains(&root) { return Ok(()); }
        if let Some(owner) = self.capture_owner(stream_id) {
            if owner == stream_id && self.capture_warmup_root(stream_id) == Some(root) { return Ok(()); }
            return Err(error("Warmup branch already belongs to another capture window"));
        }
        let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
        command.streams.current().capture = CaptureState {
            owner: Some(stream_id), root: Some(root), warming: true, ..Default::default()
        };
        drop(command);
        self.captures.warm_members.insert(stream_id, root);
        Ok(())
    }

    fn discard_capture_warmup(&mut self, root: StreamId) {
        self.streams.for_each_initialized(|stream| {
            if stream.capture.root == Some(root) && stream.capture.warming {
                stream.capture = Default::default();
            }
        });
        self.captures.warm_members.retain(|_, owner| *owner != root);
        self.captures.preparing.remove(&root);
        self.captures.communication.remove(&root);
        self.interop.end_warmup(root);
    }

    pub(super) fn capture_failure(&mut self, stream_id: StreamId, failure: &ServerError) {
        if let Some(root) = self.capture_group(stream_id) {
            self.streams.for_each_initialized(|stream| {
                if stream.capture.root == Some(root) && stream.capture.recording {
                    stream.capture.failed.get_or_insert_with(|| failure.clone());
                }
            });
        }
    }

    pub(super) fn capture_join_stream(&mut self, stream_id: StreamId, root: StreamId, event: CUevent) -> Result<(), ServerError> {
        if !self.captures.active.contains(&root) { return Err(error("Event capture has already ended")); }
        if let Some(group) = self.capture_group(stream_id) {
            if group != root { return Err(error("An event wait cannot merge independent capture graphs")); }
            let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
            return checked("cuStreamWaitEvent in capture", unsafe { cuStreamWaitEvent(command.streams.current().sys, event, 0) });
        }
        let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
        let cursor = command.streams.cursor;
        let stream = command.streams.current();
        stream.capture.primed.clear();
        stream.drop_queue.flush(|| Fence::new(stream.sys));
        stream.drop_queue.flush(|| Fence::new(stream.sys));
        checked("cuStreamSynchronize before joining capture", unsafe { cuStreamSynchronize(stream.sys) })?;
        checked("cuStreamWaitEvent join capture", unsafe { cuStreamWaitEvent(stream.sys, event, 0) })?;
        stream.capture.owner = Some(stream_id);
        stream.capture.root = Some(root);
        stream.capture.entry_cursor = cursor;
        stream.capture.recording = true;
        stream.capture.warming = false;
        command.streams.for_each_initialized(|stream| {
            let forbid = stream.capture.root == Some(root) && stream.capture.recording;
            stream.memory_management_gpu.begin_capture(root, forbid);
            stream.memory_management_cpu.begin_capture(root, forbid);
        });
        drop(command);
        self.discard_capture_warmup(stream_id);
        self.captures.warm_members.remove(&stream_id);
        self.captures.members.insert(stream_id, root);
        Ok(())
    }

    pub(super) fn capture_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.set_current_checked()?;
        if self.capture_owner(stream_id).is_some() {
            return Err(error("A capture window is already preparing or recording"));
        }
        if self.captures.active.is_empty() {
            checked("cuCtxSynchronize before capture warmup", unsafe { cuCtxSynchronize() })?;
            let mut failure = None;
            self.streams.for_each_initialized(|stream| {
                if let Err(err) = stream.memory_management_gpu.compact() { failure = Some(err); }
            });
            if let Some(failure) = failure { return Err(failure.into()); }
        }
        let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
        command.streams.current().capture = CaptureState { owner: Some(stream_id), root: Some(stream_id), warming: true, ..Default::default() };
        drop(command);
        self.captures.preparing.insert(stream_id);
        Ok(())
    }

    pub(super) fn capture_start(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        if self.captures.active.contains(&stream_id) || !self.captures.preparing.contains(&stream_id) {
            return Err(error("Capture requires a prepared, idle stream"));
        }
        let result = (|| {
            if !self.ctx.timestamps.is_empty() { return Err(error("Profiling must finish before graph capture")); }
            if let Some(communication) = self.captures.communication.get(&stream_id) {
                checked("cuStreamSynchronize collective warmup", unsafe { cuStreamSynchronize(communication.stream.raw as CUstream) })?;
            }
            let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
            let entry_cursor = command.streams.cursor;
            let stream = command.streams.current();
            stream.capture.primed.clear();
            stream.drop_queue.flush(|| Fence::new(stream.sys));
            stream.drop_queue.flush(|| Fence::new(stream.sys));
            checked("cuStreamSynchronize before capture", unsafe { cuStreamSynchronize(stream.sys) })?;
            stream.capture.entry_cursor = entry_cursor;
            stream.capture.failed = None;
            checked("cuStreamBeginCapture", unsafe {
                cuStreamBeginCapture_v2(stream.sys, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            })?;
            stream.capture.recording = true;
            stream.capture.warming = false;
            command.streams.for_each_initialized(|stream| {
                let forbid = stream.capture.root == Some(stream_id) && stream.capture.recording;
                stream.memory_management_gpu.begin_capture(stream_id, forbid);
                stream.memory_management_cpu.begin_capture(stream_id, forbid);
            });
            Ok(())
        })();
        self.captures.preparing.remove(&stream_id);
        if let Err(err) = result {
            self.discard_capture_warmup(stream_id);
            if let Ok(mut streams) = self.streams.resolve(stream_id, [].into_iter(), false) {
                streams.current().capture = Default::default();
            }
            return Err(err);
        }
        self.interop.end_warmup(stream_id);
        self.captures.active.insert(stream_id);
        Ok(())
    }

    pub(super) fn capture_stop(&mut self, stream_id: StreamId, abort: bool) -> Result<u64, ServerError> {
        let Some(owner) = self.capture_owner(stream_id) else {
            return Err(error("No recording is active on this stream"));
        };
        self.set_current_checked()?;
        if self.captures.members.contains_key(&owner) {
            return Err(error("A joined capture must end on its origin stream"));
        }
        if self.captures.warm_members.contains_key(&owner) {
            return Err(error("A warmup branch belongs to its origin capture window"));
        }
        if !self.captures.active.contains(&owner) {
            self.discard_capture_warmup(owner);
            return Err(error("The capture was prepared but never started"));
        }
        let (physical, mut state) = {
            let mut streams = self.streams.resolve(owner, [].into_iter(), false)?;
            let stream = streams.current();
            (stream.sys, core::mem::take(&mut stream.capture))
        };
        self.streams.for_each_initialized(|stream| {
            if stream.capture.root == Some(owner) && stream.capture.recording {
                state.absorb(core::mem::take(&mut stream.capture));
            }
        });
        if !abort && owner == stream_id {
            if let Some(communication) = self.captures.communication.get(&owner).filter(|state| state.joined) {
                if let Err(failure) = capture::dependency(communication.stream.raw as CUstream, physical) {
                    state.failed.get_or_insert(failure);
                }
            }
        }
        let doomed = if abort || owner != stream_id {
            Some(error(format!("Capture owned by {owner:?} was abandoned by {stream_id:?}; prior failure: {:?}", state.failed)))
        } else { state.failed };
        let executable = unsafe { capture::instantiate(physical, doomed) };
        let mut pages = Vec::new();
        self.streams.for_each_initialized(|stream| {
            pages.extend(stream.memory_management_gpu.end_capture(owner, executable.is_ok()));
            pages.extend(stream.memory_management_cpu.end_capture(owner, executable.is_ok()));
        });
        self.captures.active.remove(&owner);
        self.captures.members.retain(|_, root| *root != owner);
        let communication = self.captures.communication.remove(&owner);
        self.discard_capture_warmup(owner);
        self.interop.end_capture(owner);
        if self.captures.active.is_empty() {
            for id in core::mem::take(&mut self.captures.deferred_releases) { self.captures.graphs.remove(&id); }
        }
        let executable = match executable {
            Ok(executable) => executable,
            Err(error) => {
                capture::fail_written(state.written.into_values(), &error);
                return Err(error);
            }
        };
        let id = match self.captures.next.checked_add(1) {
            Some(id) => id,
            None => {
                unsafe { cuGraphExecDestroy(executable); }
                let error = error("Capture graph id overflow");
                capture::fail_written(state.written.into_values(), &error);
                return Err(error);
            }
        };
        self.captures.next = id;
        let _ = unsafe { cuGraphUpload(executable, physical) };
        self.captures.graphs.insert(id, Arc::new(Captured {
            executable: executable as usize, context: self.ctx.context as usize,
            stream: physical as usize, stream_id, pages, host: state.host,
            metadata: state.metadata.into_values().collect(),
            retained: state.retained,
            written: state.written.into_values().collect(),
            children: state.children,
            communication: communication.map(|state| state.stream),
        }));
        Ok(id)
    }

    pub(super) fn capture_replay(&mut self, stream_id: StreamId, id: u64) -> Result<(), ServerError> {
        self.set_current_checked()?;
        if self.capture_owner(stream_id).is_some_and(|owner| owner != stream_id) {
            return Err(error("The physical stream belongs to another logical stream's capture"));
        }
        let graph = match self.captures.graphs.get(&id).cloned() {
            Some(graph) => graph,
            None => {
                let failure = error("Unknown captured graph");
                self.capture_failure(stream_id, &failure);
                return Err(failure);
            }
        };
        if graph.stream_id != stream_id {
            let failure = error("Graph replay changed its capture stream");
            self.capture_failure(stream_id, &failure);
            return Err(failure);
        }
        let recording = self.capture_group(stream_id).is_some();
        if recording {
            let mut streams = self.streams.resolve(stream_id, [].into_iter(), false)?;
            let capture = &mut streams.current().capture;
            for binding in &graph.written {
                if let Some(memory) = binding.memory.upgrade() {
                    capture.written.entry((memory.id(), binding.start, binding.end)).or_insert_with(|| binding.clone());
                }
            }
            capture.children.push(graph.clone());
        }
        let result = checked("cuGraphLaunch", unsafe { cuGraphLaunch(graph.executable as CUgraphExec, graph.stream as CUstream) });
        match &result {
            Err(error) => {
                capture::fail_written(graph.written.iter().cloned(), error);
                self.capture_failure(stream_id, error);
            }
            Ok(()) if !recording => {
                for binding in &graph.written {
                    if let Some(memory) = binding.memory.upgrade() { memory.clear_failure(binding.start, binding.end); }
                }
            }
            Ok(()) => {}
        }
        result
    }

    pub(super) fn capture_release(&mut self, id: u64) {
        if !self.captures.active.is_empty() { self.captures.deferred_releases.push(id); }
        else { self.captures.graphs.remove(&id); }
    }
}
