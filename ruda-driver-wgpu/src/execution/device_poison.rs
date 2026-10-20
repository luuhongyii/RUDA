use ruda::runtime::server::{DevicePoison, ServerError};
use std::sync::{Arc, OnceLock};

/// wgpu reports device loss once; every stream must continue to report it.
#[derive(Clone, Debug, Default)]
pub(crate) struct PoisonWatch {
    reason: Arc<OnceLock<String>>,
}

impl PoisonWatch {
    pub(crate) fn watch(device: &wgpu::Device) -> Self {
        let watch = Self::default();
        let reason = watch.reason.clone();
        device.set_device_lost_callback(move |kind, message| {
            let _ = reason.set(format!("{kind:?}: {message}"));
            log::error!("Wgpu device lost ({kind:?}): {message}");
        });
        let reason = watch.reason.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            if reason.get().is_some() {
                log::debug!("Wgpu error after device loss: {error}");
            } else {
                panic!("Wgpu error: {error}");
            }
        }));
        watch
    }

    pub(crate) fn check(&self) -> Result<(), ServerError> {
        match self.reason.get() {
            Some(reason) => Err(DevicePoison::new(reason.clone()).into()),
            None => Ok(()),
        }
    }
}
