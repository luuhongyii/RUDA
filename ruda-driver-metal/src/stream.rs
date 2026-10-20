use crate::memory::{MetalResource, MetalStorage, Queues, io_error, wait_command};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLEvent, MTLSharedEvent,
};
use ruda::runtime::{
    logging::ServerLogger,
    memory_management::{MemoryConfiguration, MemoryManagement, MemoryManagementOptions},
    server::{Binding, ServerError},
    stream::EventStreamBackend,
};
use ruda_core::{backtrace::BackTrace, ir::MemoryDeviceProperties};
use std::{ptr::NonNull, sync::{Arc, Mutex}};

type Fault = Arc<Mutex<Option<String>>>;

fn fault_error(faults: &[Fault]) -> Result<(), ServerError> {
    for fault in faults {
        if let Some(reason) = fault.lock().unwrap_or_else(|error| error.into_inner()).clone() {
            return Err(ServerError::Generic { reason, backtrace: BackTrace::capture() });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct ActiveEncoder {
    pub command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    pub temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub resources: Vec<MetalResource>,
    pub bindings: Vec<Binding>,
}

fn completion(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    retained: (Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, Vec<MetalResource>, Vec<Binding>),
    signal: Option<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    fault: Fault,
) {
    let retained = Mutex::new(Some(retained));
    let block = block2::RcBlock::new(move |command: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
        // Metal owns the command buffer for the duration of this callback.
        let command = unsafe { command.as_ref() };
        if command.status() == MTLCommandBufferStatus::Error {
            let reason = command.error().map(|error| error.localizedDescription().to_string())
                .unwrap_or_else(|| "Metal command buffer failed".into());
            let mut fault = fault.lock().unwrap_or_else(|error| error.into_inner());
            if fault.is_none() { *fault = Some(reason); }
            if let Some((event, value)) = &signal { event.setSignaledValue(*value); }
        }
        retained.lock().unwrap_or_else(|error| error.into_inner()).take();
    });
    // Metal copies this block. Retained native objects have atomic reference counts;
    // allocator leases and retained bindings are thread-safe, and mutation is locked.
    unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&block) as *mut _); }
}

#[derive(Debug)]
pub(crate) struct MetalStream {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub memory: MemoryManagement<MetalStorage>,
    active: Option<ActiveEncoder>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    counter: u64,
    fault: Fault,
    faults: Vec<Fault>,
    pub batch_ops: usize,
    pub batch_bytes: usize,
    pub max_ops: usize,
    pub max_mb: usize,
    submitted_ops: usize,
    max_submitted_ops: usize,
    last_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pub profiling: Option<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

impl MetalStream {
    pub fn check(&self) -> Result<(), ServerError> { fault_error(&self.faults) }

    pub fn encoder(&mut self) -> Result<&mut ActiveEncoder, ServerError> {
        self.check()?;
        if self.active.is_none() {
            let command = self.queue.commandBuffer().ok_or_else(|| io_error("Metal command allocation failed"))?;
            let encoder = command.computeCommandEncoder().ok_or_else(|| io_error("Metal compute encoder allocation failed"))?;
            self.active = Some(ActiveEncoder {
                command, encoder, temporaries: Vec::new(), resources: Vec::new(), bindings: Vec::new(),
            });
        }
        Ok(self.active.as_mut().expect("encoder initialized"))
    }

    fn commit_active(&mut self, signal: Option<u64>) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, ServerError> {
        let (command, retained) = if let Some(active) = self.active.take() {
            active.encoder.endEncoding();
            (active.command, (active.temporaries, active.resources, active.bindings))
        } else {
            (self.queue.commandBuffer().ok_or_else(|| io_error("Metal event command allocation failed"))?,
             (Vec::new(), Vec::new(), Vec::new()))
        };
        let signal = signal.map(|value| (self.event.clone(), value));
        if let Some((event, value)) = &signal {
            let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&**event);
            command.encodeSignalEvent_value(event, *value);
        }
        completion(&command, retained, signal, self.fault.clone());
        command.commit();
        if self.batch_ops > 0 {
            if let Some(profiling) = &mut self.profiling { profiling.push(command.clone()); }
        }
        self.submitted_ops += self.batch_ops;
        self.batch_ops = 0;
        self.batch_bytes = 0;
        self.last_command = Some(command.clone());
        if self.submitted_ops >= self.max_submitted_ops {
            wait_command(&command)?;
            self.submitted_ops = 0;
            self.check()?;
        }
        Ok(command)
    }

    pub fn flush_checked(&mut self) -> Result<MetalEvent, ServerError> {
        self.counter = self.counter.checked_add(1).ok_or_else(|| io_error("Metal event counter exhausted"))?;
        let command = self.commit_active(Some(self.counter))?;
        Ok(MetalEvent {
            event: self.event.clone(), value: self.counter, command: Some(command),
            faults: self.faults.clone(), error: None,
        })
    }

    pub fn finish(&mut self) -> Result<(), ServerError> {
        self.flush_checked()?.wait()?;
        self.submitted_ops = 0;
        self.check()
    }

    fn wait_dependency(&mut self, dependency: MetalEvent) -> Result<(), ServerError> {
        if let Some(error) = dependency.error { return Err(error); }
        fault_error(&dependency.faults)?;
        if std::ptr::eq(&*self.event, &*dependency.event) { return Ok(()); }
        if self.active.is_some() { self.commit_active(None)?; }
        let command = self.queue.commandBuffer().ok_or_else(|| io_error("Metal wait command allocation failed"))?;
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*dependency.event);
        command.encodeWaitForEvent_value(event, dependency.value);
        completion(&command, (Vec::new(), Vec::new(), Vec::new()), None, self.fault.clone());
        command.commit();
        self.last_command = Some(command);
        for fault in dependency.faults {
            if !self.faults.iter().any(|existing| Arc::ptr_eq(existing, &fault)) {
                self.faults.push(fault);
            }
        }
        Ok(())
    }
}

impl Drop for MetalStream {
    fn drop(&mut self) {
        if let Err(error) = self.finish() { log::error!("Metal stream shutdown: {error}"); }
    }
}

#[derive(Debug)]
pub(crate) struct MetalEvent {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    faults: Vec<Fault>,
    error: Option<ServerError>,
}

// MTLSharedEvent waits and retained command-buffer completion waits are thread-safe.
unsafe impl Send for MetalEvent {}

impl MetalEvent {
    pub fn wait(self) -> Result<(), ServerError> {
        if let Some(error) = self.error { return Err(error); }
        if !self.event.waitUntilSignaledValue_timeoutMS(self.value, 60_000) {
            return Err(io_error("Metal event wait timed out").into());
        }
        if let Some(command) = self.command { wait_command(&command)?; }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        fault_error(&self.faults)
    }
}

#[derive(Debug)]
pub(crate) struct MetalStreamBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    properties: MemoryDeviceProperties,
    configuration: MemoryConfiguration,
    logger: Arc<ServerLogger>,
    queues: Arc<Queues>,
}

impl MetalStreamBackend {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>, properties: MemoryDeviceProperties,
        configuration: MemoryConfiguration, logger: Arc<ServerLogger>,
    ) -> Self {
        Self { device, properties, configuration, logger, queues: Arc::new(Queues::default()) }
    }
}

impl EventStreamBackend for MetalStreamBackend {
    type Stream = MetalStream;
    type Event = MetalEvent;

    fn create_stream(&self) -> MetalStream {
        self.try_create_stream().unwrap_or_else(|error| panic!("Metal stream creation: {error}"))
    }
    fn try_create_stream(&self) -> Result<MetalStream, ServerError> {
        let queue = self.device.newCommandQueue().ok_or_else(|| io_error("Metal queue allocation failed"))?;
        let event = self.device.newSharedEvent().ok_or_else(|| io_error("Metal shared event allocation failed"))?;
        let memory = MemoryManagement::from_configuration(
            MetalStorage::new(self.device.clone(), self.queues.clone())?, &self.properties,
            self.configuration.clone(), self.logger.clone(), MemoryManagementOptions::new("Metal GPU Memory"),
        );
        self.queues.register(queue.clone());
        let arch = self.device.architecture().name().to_string();
        let (max_ops, max_mb, max_submitted_ops) = match arch.chars().last() {
            Some('s' | 'd') => (50, 50, 512), Some('p') => (20, 20, 256), _ => (40, 40, 512),
        };
        let fault = Arc::new(Mutex::new(None));
        Ok(MetalStream {
            device: self.device.clone(), queue, memory, active: None, event, counter: 0,
            fault: fault.clone(), faults: vec![fault], batch_ops: 0, batch_bytes: 0,
            max_ops, max_mb, max_submitted_ops, submitted_ops: 0, last_command: None, profiling: None,
        })
    }
    fn handle_cursor(stream: &MetalStream, handle: &Binding) -> u64 {
        stream.memory.get_cursor(handle.memory.clone()).unwrap_or(u64::MAX)
    }
    fn is_healthy(stream: &MetalStream) -> bool { stream.check().is_ok() }
    fn flush(stream: &mut MetalStream) -> MetalEvent {
        match stream.flush_checked() {
            Ok(event) => event,
            Err(error) => MetalEvent {
                event: stream.event.clone(), value: stream.counter, command: None,
                faults: stream.faults.clone(), error: Some(error),
            },
        }
    }
    fn dependency_event_for(stream: &mut MetalStream, _: bool) -> Result<Option<MetalEvent>, ServerError> {
        stream.flush_checked().map(Some)
    }
    fn wait_event(stream: &mut MetalStream, event: MetalEvent) {
        if let Err(error) = stream.wait_dependency(event) {
            let mut fault = stream.fault.lock().unwrap_or_else(|error| error.into_inner());
            if fault.is_none() { *fault = Some(error.to_string()); }
        }
    }
    fn wait_event_checked(stream: &mut MetalStream, event: MetalEvent) -> Result<(), ServerError> {
        stream.wait_dependency(event)
    }
    fn wait_event_sync(event: MetalEvent) -> Result<(), ServerError> { event.wait() }
}

