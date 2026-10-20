use crate::{Backend, BackendGraph, StreamId};

/// A closure recorded as a hardware graph, or rerun on unsupported backends.
pub struct CapturedGraph<B: Backend, T, F> {
    device: B::Device,
    stream: StreamId,
    output: T,
    closure: F,
    hardware: Option<BackendGraph>,
}

struct CaptureGuard<'a, B: Backend> {
    device: &'a B::Device,
    stream: StreamId,
    active: bool,
}

impl<B: Backend> Drop for CaptureGuard<'_, B> {
    fn drop(&mut self) {
        if self.active { self.stream.executes(|| B::graph_abort_capture(self.device)); }
    }
}

/// Warm up three times, then capture a closure using stable device buffers.
/// Hardware capture retains the recorded pages; unsupported backends rerun the closure.
pub fn capture<B: Backend, T, F: FnMut() -> T>(device: &B::Device, mut closure: F) -> CapturedGraph<B, T, F> {
    let stream = StreamId::current();
    let prepared = stream.executes(|| B::graph_prepare(device)).is_ok();
    let mut guard = CaptureGuard::<B> { device, stream, active: prepared };
    for _ in 0..3 {
        let output = stream.executes(&mut closure);
        let _ = stream.executes(|| B::sync(device));
        drop(output);
    }
    let (hardware, output) = if !prepared {
        (None, stream.executes(&mut closure))
    } else { match stream.executes(|| B::graph_start_capture(device)) {
        Ok(()) => {
            let output = stream.executes(&mut closure);
            let hardware = stream.executes(|| B::graph_stop_capture(device)).ok();
            if hardware.is_none() { stream.executes(|| B::graph_abort_capture(device)); }
            guard.active = false;
            (hardware, output)
        }
        Err(_) => {
            stream.executes(|| B::graph_abort_capture(device));
            guard.active = false;
            (None, stream.executes(&mut closure))
        }
    } };
    CapturedGraph { device: device.clone(), stream, output, closure, hardware }
}

impl<B: Backend, T, F: FnMut() -> T> CapturedGraph<B, T, F> {
    /// Replay the recorded work, preserving output buffers on the hardware path.
    ///
    /// # Safety
    /// Captured inputs must remain live and stable; refresh inputs and consume outputs
    /// on the capture stream, and synchronize accesses from any other stream or thread.
    pub unsafe fn replay(&mut self) -> &T {
        self.stream.executes(|| match &self.hardware {
            Some(graph) => unsafe { B::graph_replay(&self.device, graph) }.expect("Graph replay failed"),
            None => self.output = (self.closure)(),
        });
        &self.output
    }

    /// The output buffers. Hardware outputs become valid after the first replay.
    pub fn output(&self) -> &T { &self.output }

    /// Whether replay dispatches a hardware executable rather than rerunning Rust.
    pub fn is_hardware(&self) -> bool { self.hardware.is_some() }
}
