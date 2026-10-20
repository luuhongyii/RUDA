use ruda_autodiff::{Autodiff, checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing}};
use ruda_model::tensor::{backend::BackendTypes, Tensor};

macro_rules! tape_suite {
    ($name:ident, $strategy:ty) => {
        mod $name {
            type AD = Autodiff<crate::TestBackend, $strategy>;
            type TestTensor<const D: usize> = Tensor<AD, D>;
            struct AutodiffDevice;
            impl AutodiffDevice {
                fn new() -> <AD as BackendTypes>::Device { Default::default() }
            }
            include!("tape_cases.rs");

            #[test]
            fn untracked_backward_preserves_fresh_tracked_work() {
                let device = AutodiffDevice::new();
                let leaf = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
                let fresh = leaf.clone() * leaf.clone();
                for value in [TestTensor::<1>::ones([2], &device), TestTensor::<1>::ones([2], &device).sum()] {
                    let failure = catch_unwind(AssertUnwindSafe(|| value.backward()));
                    assert!(failure.is_err());
                }
                let grads = fresh.sum().backward();
                leaf.grad(&grads).unwrap().into_data()
                    .assert_eq(&TensorData::from([4.0f32, 6.0]), false);
            }

            #[test]
            fn consumed_leaf_root_is_not_a_retained_tape_but_remains_a_gradient_endpoint() {
                let device = AutodiffDevice::new();
                let leaf = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
                let grads = leaf.backward();
                leaf.grad(&grads).unwrap().into_data()
                    .assert_eq(&TensorData::from([1.0f32, 1.0]), false);
                assert_consumed(|| { let _ = leaf.backward(); });
                let grads = leaf.clone().mul_scalar(3.0).sum().backward();
                leaf.grad(&grads).unwrap().into_data()
                    .assert_eq(&TensorData::from([3.0f32, 3.0]), false);
            }
        }
    };
}

tape_suite!(plain, NoCheckpointing);
tape_suite!(balanced, BalancedCheckpointing);
