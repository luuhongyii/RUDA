use alloc::vec::Vec;
use ruda_core::kernel::{BufferIOAttr, KernelIo};

use super::{Binding, KernelArguments, RudaCount, ServerError, WriteScope};
use crate::runtime::id::KernelId;

/// Owns producer bindings until submission completes, fails, or is abandoned.
#[derive(Debug)]
pub struct KernelExecution {
    kernel: KernelId,
    buffers: Vec<Binding>,
    tensor_maps: Vec<Binding>,
    declared_io: KernelIo,
    count: Option<Binding>,
    reads: Vec<Binding>,
    writes: Vec<Binding>,
    write_scope: Option<WriteScope>,
    armed: bool,
}

impl KernelExecution {
    /// Before compilation, use launch declarations without tainting declared read-only inputs.
    pub fn new(kernel: KernelId, args: &KernelArguments, count: &RudaCount) -> Self {
        Self::from_declared_bindings(
            kernel,
            args.buffers.clone(),
            args.tensor_maps.iter().map(|map| map.binding.clone()).collect(),
            args.declared_io.clone(),
            match count { RudaCount::Dynamic(binding) => Some(binding.clone()), _ => None },
        )
    }

    /// Create a producer scope when resources have already been prepared.
    pub fn from_bindings(
        kernel: KernelId,
        buffers: Vec<Binding>,
        tensor_maps: Vec<Binding>,
        count: Option<Binding>,
    ) -> Self {
        Self::from_declared_bindings(kernel, buffers, tensor_maps, KernelIo::default(), count)
    }

    /// Preserve launch declarations after resources have been prepared.
    pub fn from_declared_bindings(
        kernel: KernelId,
        buffers: Vec<Binding>,
        tensor_maps: Vec<Binding>,
        declared_io: KernelIo,
        count: Option<Binding>,
    ) -> Self {
        let mut execution = Self {
            kernel,
            buffers,
            tensor_maps,
            declared_io,
            count,
            reads: Vec::new(),
            writes: Vec::new(),
            write_scope: None,
            armed: true,
        };
        execution.set_io(None);
        execution
    }

    /// Refine access information after compilation or a cache hit.
    pub fn set_io(&mut self, io: Option<&KernelIo>) {
        assert!(self.write_scope.is_none(), "Cannot change I/O after entering a write scope");
        self.reads.clear();
        self.writes.clear();
        for (bindings, accesses, declared) in [
            (&self.buffers, io.map(|io| &io.buffers), &self.declared_io.buffers),
            (&self.tensor_maps, io.map(|io| &io.tensor_maps), &self.declared_io.tensor_maps),
        ] {
            for (index, binding) in bindings.iter().enumerate() {
                let access = accesses.and_then(|accesses| accesses.get(index))
                    .or_else(|| declared.get(index))
                    .copied().unwrap_or(BufferIOAttr::ReadWrite);
                if access.reads() { self.reads.push(binding.clone()); }
                if access.writes() { self.writes.push(binding.clone()); }
            }
        }
        self.reads.extend(self.count.iter().cloned());
    }

    /// Bindings consumed by the compiled kernel, including an indirect grid.
    pub fn reads(&self) -> &[Binding] { &self.reads }

    /// Bindings produced by the compiled kernel.
    pub fn writes(&self) -> &[Binding] {
        self.write_scope.as_ref().map_or(self.writes.as_slice(), WriteScope::bindings)
    }

    /// Propagate a failed dependency without creating a second root cause.
    pub fn check(&mut self) -> Result<(), ServerError> {
        assert!(self.write_scope.is_none(), "Inputs must be checked before claiming outputs");
        if let Some(error) = Binding::propagate_failure(self.kernel.clone(), self.reads.iter(), &self.writes) {
            self.armed = false;
            return Err(error);
        }
        if let Err(error) = Binding::check_all(self.reads.iter()) {
            return Err(self.fail(error));
        }
        Ok(())
    }

    /// Enter after checking inputs and resolving an empty or compile-only dispatch.
    pub fn enter(&mut self) {
        if self.armed && self.write_scope.is_none() {
            self.write_scope = Some(WriteScope::new(core::mem::take(&mut self.writes)));
        }
    }

    /// Transfer write settlement to the submitter; the queued value only retains bindings.
    /// The returned scope must complete after registration, not after executing the task.
    pub fn begin_submission(&mut self) -> Result<Option<WriteScope>, ServerError> {
        if !self.armed { return Ok(None); }
        self.check()?;
        self.enter();
        self.armed = false;
        Ok(self.write_scope.take())
    }

    /// Attribute one failure root to all outputs.
    pub fn fail(&mut self, error: ServerError) -> ServerError {
        if self.armed {
            self.enter();
            self.armed = false;
            if let Some(scope) = self.write_scope.take() {
                return scope.fail(error);
            }
        }
        error
    }

    /// Settle successful submission without clearing a later producer's claims.
    pub fn complete(mut self) {
        if self.armed {
            self.enter();
            if let Some(scope) = self.write_scope.take() { scope.complete(); }
        }
        self.armed = false;
    }

    /// Mark a dispatch as non-producing without consuming its scope.
    pub fn skip(&mut self) { self.armed = false; }

    /// A compile-only or empty dispatch did not produce any payload.
    pub fn cancel(mut self) { self.armed = false; }

    /// Finish a synchronous submission path, preserving its actual error.
    pub fn finish(mut self, result: Result<(), ServerError>) -> Result<(), ServerError> {
        match result {
            Ok(()) => { self.complete(); Ok(()) }
            Err(error) => Err(self.fail(error)),
        }
    }
}

impl Drop for KernelExecution {
    fn drop(&mut self) {
        if self.armed && self.write_scope.is_none() {
            Binding::fail_all(self.writes.iter(), ServerError::TornDown);
        }
    }
}
