use ruda_tensor::{AutodiffContext, DeviceOps};
use crate::{DispatchDevice, DispatchTensor};

#[cfg(feature = "std")]
std::thread_local! {
    static SCOPES: core::cell::RefCell<alloc::vec::Vec<(DispatchDevice, AutodiffContext)>> =
        const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
}

pub(crate) fn current(device: &DispatchDevice) -> AutodiffContext {
    #[cfg(feature = "std")]
    if let Some(context) = SCOPES.with(|scopes| scopes.borrow().iter().rev()
        .find(|(key, _)| key == device.inner()).map(|(_, context)| *context)) {
        return context;
    }
    device.context()
}

pub(crate) fn run<R>(device: &DispatchDevice, context: AutodiffContext, func: impl FnOnce() -> R) -> R {
    #[cfg(feature = "std")]
    {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) { SCOPES.with(|scopes| { scopes.borrow_mut().pop(); }); }
        }
        SCOPES.with(|scopes| scopes.borrow_mut().push((device.inner().clone(), context)));
        let _guard = Guard;
        func()
    }
    #[cfg(not(feature = "std"))]
    {
        assert_eq!(current(device), context, "Use an explicitly associated DispatchDevice without std");
        func()
    }
}

pub(crate) struct Selection {
    pub device: DispatchDevice,
    pub context: AutodiffContext,
}

impl Selection {
    pub fn factory(device: &DispatchDevice) -> Self {
        Self { device: device.inner().clone(), context: current(device) }
    }

    pub fn tensors<'a>(tensors: impl IntoIterator<Item = &'a DispatchTensor>) -> Self {
        let mut tensors = tensors.into_iter();
        let first = tensors.next().expect("A tensor operation requires at least one tensor");
        let mut selection = Self { device: first.device().inner().clone(), context: first.context() };
        for tensor in tensors {
            assert_eq!(selection.device, tensor.device().inner().clone(), "Operation inputs must use the same native device");
            selection.context = selection.context.merge(tensor.context());
        }
        selection
    }
}
