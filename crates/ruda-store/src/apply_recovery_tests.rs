extern crate std;

use crate::{ModuleAdapter, ModuleSnapshot, TensorSnapshot};
use ruda_model::module::{Module, Param, ParamId};
use ruda_tensor::api::{Bool, Tensor, TensorData, backend::Backend};
use std::{panic::{AssertUnwindSafe, catch_unwind}, sync::{Arc, atomic::{AtomicUsize, Ordering}}};
use alloc::{boxed::Box, vec};

type TestBackend = ruda_tensor_host::Host;

#[derive(Module, Debug)]
struct RecoveryModule<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    flag: Param<Tensor<B, 1, Bool>>,
}

#[derive(Clone)]
struct PanicOnSecond(Arc<AtomicUsize>);

impl ModuleAdapter for PanicOnSecond {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if self.0.fetch_add(1, Ordering::SeqCst) == 1 {
            panic!("adapter failed after mapping weight");
        }
        snapshot.clone()
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[test]
fn apply_panic_preserves_original_and_can_retry() {
    let device = Default::default();
    let mut module = RecoveryModule::<TestBackend> {
        weight: Param::from_data([2f32, 3.], &device),
        flag: Param::uninitialized(ParamId::new(),
            |device, _| Tensor::from_data([false, true], device), device, false, [2].into()),
    };
    let ids = (module.weight.id, module.flag.id);
    let loaded = Arc::new(AtomicUsize::new(0));
    let snapshots = vec![
        TensorSnapshot::from_data(TensorData::from([20f32, 30.]), vec!["weight".into()], vec![], ids.0),
        TensorSnapshot::from_data(TensorData::from([true, false]), vec!["flag".into()], vec![], ids.1),
    ];
    assert!(catch_unwind(AssertUnwindSafe(|| module.apply(
        snapshots, None, Some(Box::new(PanicOnSecond(loaded.clone()))), false,
    ))).is_err());
    assert_eq!(loaded.load(Ordering::SeqCst), 2);
    assert_eq!((module.weight.id, module.flag.id), ids);
    assert!(!module.flag.is_initialized());
    module.weight.val().into_data().assert_eq(&TensorData::from([2f32, 3.]), true);

    let snapshots = vec![
        TensorSnapshot::from_data(TensorData::from([20f32, 30.]), vec!["weight".into()], vec![], ids.0),
        TensorSnapshot::from_data(TensorData::from([true, false]), vec!["flag".into()], vec![], ids.1),
    ];
    let result = module.apply(snapshots, None, None, false);
    assert!(result.errors.is_empty());
    assert_eq!(result.applied.len(), 2);
    assert_eq!((module.weight.id, module.flag.id), ids);
    module.weight.val().into_data().assert_eq(&TensorData::from([20f32, 30.]), true);
    module.flag.val().into_data().assert_eq(&TensorData::from([true, false]), false);
}
