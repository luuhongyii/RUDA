use crate::memory::manager::WgpuMemManager;
use super::{poll::WgpuPoll, timings::{QueryProfiler, TimestampQuerySetBudget, TimestampSampling}};
use crate::{WgpuResource, memory::controller::WgpuAllocController, schedule::ScheduleTask};
use ruda_core::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use ruda_kernel::dsl::{
    RudaCount, MemoryConfiguration,
    future::{self, DynFut},
    server::{IoError, ProfileError, ProfilingToken, ServerError, StreamErrorMode},
    zspace::Shape,
};
use ruda_core::ir::MemoryDeviceProperties;
use ruda::runtime::{
    logging::ServerLogger, memory_management::ManagedMemoryHandle,
    timestamp_profiler::TimestampProfiler,
};
use std::{future::Future, num::NonZero, pin::Pin, sync::Arc};
use wgpu::ComputePipeline;

#[derive(Debug)]
enum Timings {
    Unclaimed { budget: Arc<TimestampQuerySetBudget>, sampling: TimestampSampling },
    Device(Box<QueryProfiler>),
    System(TimestampProfiler),
}

impl Timings {
    fn system() -> Self {
        if cfg!(target_family = "wasm") {
            panic!("Cannot profile on web assembly without timestamp_query feature as it requires blocking.");
        }
        Self::System(TimestampProfiler::default())
    }
}

#[derive(Debug)]
pub struct WgpuStream {
    pub mem_manage: WgpuMemManager,
    pub device: wgpu::Device,
    poison: super::device_poison::PoisonWatch,
    compute_pass: Option<wgpu::ComputePass<'static>>,
    timings: Timings,
    tasks_count: usize,
    tasks_max: usize,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    poll: WgpuPoll,
    submission_load: SubmissionLoad,
    /// Number of consecutive `write_buffer` calls without a `queue.submit()`.
    /// Used to prevent wgpu staging buffer pool exhaustion during bulk writes
    /// (e.g. model loading with hundreds of tensors).
    pending_write_count: usize,
    pending_address_pins: Vec<ruda::runtime::memory_management::MemoryResourcePin>,
}

impl WgpuStream {
    /// Creates a new WGPU stream.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        timing_budget: Arc<TimestampQuerySetBudget>,
        timestamp_sampling: TimestampSampling,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
        poison: super::device_poison::PoisonWatch,
    ) -> Self {
        let timings = if timing_method == TimingMethod::Device {
            Timings::Unclaimed { budget: timing_budget, sampling: timestamp_sampling }
        } else {
            Timings::system()
        };

        let poll = WgpuPoll::new(device.clone());

        #[allow(unused_mut)]
        let mut mem_manage =
            WgpuMemManager::new(device.clone(), queue.clone(), memory_properties, memory_config, logger);

        Self {
            mem_manage,
            compute_pass: None,
            timings,
            poison,
            encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Ruda Tasks Encoder"),
                })
            },
            device,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            submission_load: SubmissionLoad::default(),
            pending_write_count: 0,
            pending_address_pins: Vec::new(),
        }
    }

    /// Enqueue a [`ScheduleTask`] on this stream.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute.
    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer, destination: _destination } => {
                // It is important to flush before writing, as the write operation is inserted
                // into the QUEUE not the encoder. We want to make sure all outstanding work
                // happens _before_ the write operation.
                if let Err(error) = self.flush(StreamErrorMode { ignore: true, flush: false }) {
                    self.error(error);
                    return;
                }
                self.write_to_buffer(&buffer, &data);
            }
            ScheduleTask::Execute {
                pipeline,
                count,
                count_resource,
                resources,
                execution,
            } => {
                if let Err(error) = self.poison.check() {
                    self.error(error);
                    return;
                }
                let resources = resources.into_resources(self);
                match self.register_pipeline(pipeline, resources.iter(), &count, count_resource.as_ref()) {
                    Ok(()) => execution.complete(),
                    Err(error) => {
                        self.error(error);
                    }
                }
            }
        }
    }

    pub fn sync(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + 'static>> {
        if let Err(error) = self.poison.check() { return Box::pin(async move { Err(error) }); }
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);

        let flush_error = self
            .flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .err();

        let queue = self.queue.clone();
        let error_future = error_scope.pop();
        let poll = self.poll.start_polling();
        let poison = self.poison.clone();

        Box::pin(async move {
            let (sender, receiver) = async_channel::bounded::<()>(1);
            queue.on_submitted_work_done(move || {
                // Signal that we're done.
                let _ = sender.try_send(());
                core::mem::drop(poll);
            });
            let _ = receiver.recv().await;
            poison.check()?;

            let scoped_error = error_future.await;
            poison.check()?;
            if let Some(error) = scoped_error {
                return Err(ServerError::Generic {
                    reason: format!("{error}"),
                    backtrace: BackTrace::capture(),
                });
            }

            match flush_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.poison.check().map_err(|error| IoError::Execution(Box::new(error)))?;
        if self.mem_manage.relocation_pending(size) {
            self.flush(StreamErrorMode { ignore: false, flush: false })
                .map_err(|error| IoError::Execution(Box::new(error)))?;
        }
        self.mem_manage.reserve(size)
    }

    pub(crate) fn check_device(&self) -> Result<(), ServerError> { self.poison.check() }

    /// Invalidate active measurements; buffers, not the stream, carry ordinary failures.
    pub fn error(&mut self, error: ServerError) {
        self.profile_error(ProfileError::Server(Box::new(error.clone())));
    }

    pub(crate) fn create_uniform(&mut self, data: &[u8]) -> WgpuResource {
        let resource = self.mem_manage.reserve_uniform(data.len() as u64);
        self.write_to_buffer(&resource, data);
        resource
    }

    // Nb: this function submits a command to the _queue_ not to the encoder,
    // so you have to be really careful about the ordering of operations here.
    // Any buffer which has outstanding (not yet flushed) compute work should
    // NOT be copied to.
    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max {
            let _ = self
                .flush(StreamErrorMode {
                    ignore: true,
                    flush: false,
                })
                .ok();
        }
    }

    pub fn flush(&mut self, _mode: StreamErrorMode) -> Result<(), ServerError> {
        self.poison.check()?;
        if self.tasks_count == 0 {
            return Ok(());
        }

        // End the current compute pass.
        self.compute_pass = None;

        // Submit the pending actions to the queue. This will _first_ submit the
        // pending uniforms copy operations, then the main tasks.
        let tasks_encoder = {
            std::mem::replace(&mut self.encoder, {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Ruda Tasks Encoder"),
                    })
            })
        };

        // This will _first_ fire off all pending write_buffer work.
        let index = self.queue.submit([tasks_encoder.finish()]);
        // Unsubmitted encoders on any stream keep their resolved addresses
        // fixed. Submitted work is covered by the relocation completion wait.
        self.pending_address_pins.clear();

        self.submission_load
            .regulate(&self.device, self.tasks_count, index);

        // Cleanup allocations and deallocations.
        self.mem_manage.memory_cleanup(false);
        self.mem_manage.release_uniforms();

        self.tasks_count = 0;
        self.pending_write_count = 0;

        self.poison.check()
    }

    fn register_pipeline<'a>(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: impl Iterator<Item = &'a WgpuResource>,
        dispatch: &RudaCount,
        count_resource: Option<&WgpuResource>,
    ) -> Result<(), ServerError> {
        if dispatch.is_empty() {
            return Ok(());
        }

        let entries = resources
            .enumerate()
            .map(|(i, r)| {
                if let Some(pin) = r.address_pin() { self.pending_address_pins.push(pin); }
                wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: r.as_wgpu_bind_resource(),
                }
            })
            .collect::<Vec<_>>();

        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.compute_pass.get_or_insert_with(|| {
            let writes = if let Timings::Device(query_time) = &mut self.timings {
                query_time.register_profile_device(&self.device)
            } else {
                None
            };
            self.encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: writes,
                })
                .forget_lifetime()
        });

        self.tasks_count += 1;

        let group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &entries,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        match dispatch.clone() {
            RudaCount::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            RudaCount::Dynamic(_) => {
                let res = count_resource.ok_or_else(|| IoError::NotFound {
                    reason: "Missing indirect dispatch buffer".into(),
                    backtrace: BackTrace::capture(),
                })?;
                if let Some(pin) = res.address_pin() { self.pending_address_pins.push(pin); }
                pass.dispatch_workgroups_indirect(&res.buffer, res.offset);
            }
        }
        self.flush_if_needed();
        Ok(())
    }

}

mod io;
mod profiling;
mod submission;
use submission::*;
