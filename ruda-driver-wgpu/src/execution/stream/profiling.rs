use super::*;

impl WgpuStream {
    fn claim_timings(&mut self) {
        let Timings::Unclaimed { budget, sampling } = &self.timings else { return; };
        self.timings = if budget.try_acquire() {
            Timings::Device(Box::new(QueryProfiler::new(&self.queue, &self.device, budget.clone(), *sampling)))
        } else {
            Timings::system()
        };
    }

    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        self.poison.check()?;
        self.claim_timings();
        match &self.timings {
            Timings::System(_) => ruda_core::future::block_on(self.sync())?,
            Timings::Device(query) if query.sampling().drains_before_window() => {
                self.flush(StreamErrorMode { ignore: false, flush: true })?;
                if let Err(error) = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }) {
                    log::warn!("waiting for work ahead of a profiled window: {error}");
                }
            }
            Timings::Device(_) | Timings::Unclaimed { .. } => {
                self.flush(StreamErrorMode { ignore: false, flush: true })?;
            }
        }

        self.poison.check()?;
        match &mut self.timings {
            Timings::Unclaimed { .. } => unreachable!("claimed above"),
            Timings::System(_) => {
                let profiler = self.system_profiler();
                Ok(profiler.start())
            }
            Timings::Device(query) => {
                self.compute_pass = None;
                let token = query.start_profile();
                Ok(token)
            }
        }
    }

    pub fn profile_error(&mut self, error: ProfileError) {
        match &mut self.timings {
            Timings::Unclaimed { .. } => {}
            Timings::Device(profiler) => {
                profiler.error(error);
            }
            Timings::System(profiler) => {
                profiler.error(error);
            }
        }
    }

    pub fn abandon_profile(&mut self, token: ProfilingToken) {
        match &mut self.timings {
            Timings::Unclaimed { .. } => {}
            Timings::Device(profiler) => profiler.abandon_profile(token),
            Timings::System(profiler) => profiler.abandon(token),
        }
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        if let Err(error) = self.poison.check() {
            self.abandon_profile(token);
            return Err(ProfileError::Server(Box::new(error)));
        }
        match &mut self.timings {
            Timings::Unclaimed { .. } => Err(ProfileError::NotRegistered { backtrace: BackTrace::capture() }),
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                let result = future::block_on(self.sync());
                let profiler = self.system_profiler();

                if let Err(err) = result {
                    profiler.error(ProfileError::Server(Box::new(err)));
                }
                profiler.stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();
                self.compute_pass = None;

                // Submit commands needed for profiling.
                let readback = {
                    let Timings::Device(timing) = &mut self.timings else {
                        return Err(ProfileError::Unknown {
                            reason: "Unexpected timings type".to_string(),
                            backtrace: BackTrace::capture(),
                        });
                    };
                    timing.stop_profile_setup(token, &self.device)?
                };

                // This flushes the queue to execute the encoder write command to write the
                // timings.
                self.tasks_count += 1;
                let result = self.flush(StreamErrorMode {
                    ignore: false,
                    flush: true,
                });

                let Timings::Device(timing) = &mut self.timings else {
                    return Err(ProfileError::Unknown {
                        reason: "Unexpected timings type".to_string(),
                        backtrace: BackTrace::capture(),
                    });
                };

                match result {
                    Ok(_) => {
                        let map_buffer = readback.map(|readback| {
                            if timing.sampling().resolves_after_completion() {
                                if let Err(error) = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }) {
                                    log::warn!("waiting for a profiled window to complete: {error}");
                                }
                            }
                            self.queue.submit([readback.commands]);
                            readback.map_buffer
                        });
                        let result = timing.stop_profile(map_buffer, poll);
                        self.poison.check().map_err(|error| ProfileError::Server(Box::new(error)))?;
                        result
                    }
                    Err(err) => {
                        drop(readback);
                        drop(poll);
                        Err(ProfileError::Server(Box::new(err)))
                    }
                }
            }
        }
    }
}
