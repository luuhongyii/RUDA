use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};
use ruda_tensor::{AutodiffContext, Backend};

pub(crate) fn current<B: Backend, C: CheckpointStrategy>(device: &B::Device) -> AutodiffContext {
    scoped::<B, C>(device).unwrap_or(AutodiffContext::Enabled(C::policy()))
}

pub(crate) fn scoped<B: Backend, C: CheckpointStrategy>(device: &B::Device) -> Option<AutodiffContext> {
    #[cfg(feature = "std")]
    { local::get::<Autodiff<B, C>>(device) }
    #[cfg(not(feature = "std"))]
    { let _ = device; None }
}

pub(crate) fn run<B: Backend, C: CheckpointStrategy, R>(device: &B::Device, context: AutodiffContext, func: impl FnOnce() -> R) -> R {
    if let AutodiffContext::Enabled(policy @ ruda_tensor::GradientCheckpointingStrategy::Custom(_)) = context {
        assert_eq!(policy, C::policy(), "Checkpointing strategy is not provided by this backend");
    }
    #[cfg(feature = "std")]
    { local::run::<Autodiff<B, C>, R>(device, context, func) }
    #[cfg(not(feature = "std"))]
    {
        assert_eq!(current::<B, C>(device), context, "Scoped creation context requires std; use explicit per-tensor context in no_std");
        func()
    }
}

#[cfg(feature = "std")]
mod local {
    use super::*;
    use core::{any::TypeId, cell::RefCell};
    use ruda_tensor::{DeviceId, DeviceOps};

    std::thread_local! {
        static SCOPES: RefCell<alloc::vec::Vec<(TypeId, DeviceId, AutodiffContext)>> = const { RefCell::new(alloc::vec::Vec::new()) };
    }

    pub(super) fn get<B: Backend>(device: &B::Device) -> Option<AutodiffContext> {
        let key = (TypeId::of::<B>(), device.id());
        SCOPES.with(|scopes| scopes.borrow().iter().rev()
            .find(|(backend, device, _)| (*backend, *device) == key).map(|entry| entry.2))
    }

    pub(super) fn run<B: Backend, R>(device: &B::Device, context: AutodiffContext, func: impl FnOnce() -> R) -> R {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) { SCOPES.with(|scopes| { scopes.borrow_mut().pop(); }); }
        }
        SCOPES.with(|scopes| scopes.borrow_mut().push((TypeId::of::<B>(), device.id(), context)));
        let guard = Guard;
        let output = func();
        drop(guard);
        output
    }
}
