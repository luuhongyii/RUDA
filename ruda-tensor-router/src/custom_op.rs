use alloc::{string::{String, ToString}, sync::Arc};
use hashbrown::HashMap;
use ruda_tensor::{BackendTypes, graph::{BackendIr, CustomOpIr, HandleContainer}};

/// Backend operation receiving tensor handles, scalar arguments and the runner device.
pub type CustomOpHandler<B> = Arc<dyn Fn(
    &mut HandleContainer<<B as BackendIr>::Handle>,
    &CustomOpIr,
    &<B as BackendTypes>::Device,
) + Send + Sync>;

/// Owned custom-operation handlers, shared read-only by cloned runners.
#[derive(Clone)]
pub struct CustomOpRegistry<B: BackendIr> {
    handlers: HashMap<String, CustomOpHandler<B>>,
}

impl<B: BackendIr> Default for CustomOpRegistry<B> {
    fn default() -> Self { Self { handlers: HashMap::new() } }
}

impl<B: BackendIr> CustomOpRegistry<B> {
    /// Create an empty registry.
    pub fn new() -> Self { Self::default() }

    /// Register a handler; registering the same ID replaces its previous handler.
    pub fn register<F>(&mut self, id: &str, handler: F)
    where F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static {
        self.handlers.insert(id.to_string(), Arc::new(handler));
    }

    pub(crate) fn get(&self, id: &str) -> Option<&CustomOpHandler<B>> { self.handlers.get(id) }
}
