use crate::{Autodiff, checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing}};
use alloc::{vec, vec::Vec};
use ruda_tensor::{DType, TensorData, api::{Int, Tensor}, tensor::IndexingUpdateOp};

type Inner = ruda_tensor_host::Host;

fn updates() -> [IndexingUpdateOp; 5] {
    [IndexingUpdateOp::Assign, IndexingUpdateOp::Add, IndexingUpdateOp::Mul,
        IndexingUpdateOp::Min, IndexingUpdateOp::Max]
}

fn fold(a: f32, b: f32, update: IndexingUpdateOp) -> f32 {
    match update {
        IndexingUpdateOp::Assign => b,
        IndexingUpdateOp::Add => a + b,
        IndexingUpdateOp::Mul => a * b,
        IndexingUpdateOp::Min => a.min(b),
        IndexingUpdateOp::Max => a.max(b),
    }
}

fn destination(flat: usize, shape: [usize; 2], input: [usize; 2], axis: usize, index: usize) -> usize {
    let mut coord = [flat / shape[1], flat % shape[1]];
    coord[axis] = index;
    coord[0] * input[1] + coord[1]
}

#[test]
fn numeric_updates_cover_axes_duplicate_indices_and_strided_inputs() {
    let device = Default::default();
    for strided in [false, true] {
        let (shape, data) = if strided {
            ([3, 2], vec![2f32, 7., 5., 1., 3., 4.])
        } else {
            ([2, 3], vec![2f32, 5., 3., 7., 1., 4.])
        };
        for axis in 0..2 {
            let indices = [shape[axis] - 1, 0, shape[axis] - 1, 1, 0];
            let mut value_shape = shape;
            value_shape[axis] = indices.len();
            let values: Vec<f32> = (0..value_shape[0] * value_shape[1]).map(|i| (i % 5) as f32 - 2.).collect();
            let scatter: Vec<i64> = (0..values.len()).map(|i| indices[if axis == 0 { i / value_shape[1] } else { i % value_shape[1] }] as i64).collect();
            for update in updates() {
                let mut expected = data.clone();
                for (i, &value) in values.iter().enumerate() {
                    let target = destination(i, value_shape, shape, axis, scatter[i] as usize);
                    expected[target] = fold(expected[target], value, update);
                }
                for select in [false, true] {
                    let tensor = Tensor::<Inner, 2>::from_data([[2., 5., 3.], [7., 1., 4.]], &device);
                    let tensor = if strided { tensor.swap_dims(0, 1) } else { tensor };
                    let value = Tensor::<Inner, 2>::from_data(TensorData::new(values.clone(), value_shape), &device);
                    let output = if select {
                        tensor.select_assign(axis, Tensor::from_data(indices.map(|i| i as i32), &device), value, update)
                    } else {
                        tensor.scatter(axis, Tensor::from_data(TensorData::new(scatter.clone(), value_shape), &device), value, update)
                    };
                    output.into_data().assert_eq(&TensorData::new(expected.clone(), shape), false);
                    let tensor = Tensor::<Inner, 2, Int>::from_data([[2, 5, 3], [7, 1, 4]], &device);
                    let tensor = if strided { tensor.swap_dims(0, 1) } else { tensor };
                    let value = Tensor::<Inner, 2, Int>::from_data(TensorData::new(values.iter().map(|x| *x as i32).collect(), value_shape), &device);
                    let output = if select {
                        tensor.select_assign(axis, Tensor::from_data(indices.map(|i| i as i32), &device), value, update)
                    } else {
                        tensor.scatter(axis, Tensor::from_data(TensorData::new(scatter.clone(), value_shape), &device), value, update)
                    };
                    output.into_data().assert_eq(&TensorData::new(expected.iter().map(|x| *x as i32).collect(), shape), false);
                }
            }
        }
    }
}

#[test]
fn selected_i64_values_preserve_large_integers() {
    let device = Default::default();
    for update in updates() {
        let data = [9_007_199_254_740_993i64, -9_007_199_254_740_993, 11];
        let values = [2i64, 3];
        let mut expected = data;
        for (&index, &value) in [1usize, 0].iter().zip(values.iter()) {
            expected[index] = match update {
                IndexingUpdateOp::Assign => value,
                IndexingUpdateOp::Add => expected[index] + value,
                IndexingUpdateOp::Mul => expected[index] * value,
                IndexingUpdateOp::Min => expected[index].min(value),
                IndexingUpdateOp::Max => expected[index].max(value),
            };
        }
        let output = Tensor::<Inner, 1, Int>::from_data(data, (&device, DType::I64)).select_assign(
            0, Tensor::from_data([1, 0], &device),
            Tensor::from_data(values, (&device, DType::I64)), update);
        output.into_data().assert_eq(&TensorData::from(expected), true);
    }
}

#[test]
fn selected_low_precision_values_preserve_storage_dtype() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16] {
        let data = TensorData::from([1.25f32, -2.5, 3.75]).convert_dtype(dtype);
        let values = TensorData::from([0.5f32, 2.]).convert_dtype(dtype);
        let expected = TensorData::from([2f32, -2.5, 0.5]).convert_dtype(dtype);
        let output = Tensor::<Inner, 1>::from_data(data, (&device, dtype)).select_assign(
            0, Tensor::from_data([2, 0], &device),
            Tensor::from_data(values, (&device, dtype)), IndexingUpdateOp::Assign);
        output.into_data().assert_eq(&expected, true);
    }
}

fn gradients<C: CheckpointStrategy>() {
    type A<C> = Autodiff<Inner, C>;
    let device = Default::default();
    let data = vec![2f32, 5., 3., 7., 1., 4.];
    for axis in 0..2 {
        let indices = if axis == 0 { [1usize, 0] } else { [2usize, 0] };
        let shape = [2, 3];
        let mut value_shape = shape;
        value_shape[axis] = 2;
        let values: Vec<f32> = [3f32, 5., 0., 4., 2., 7.][..value_shape[0] * value_shape[1]].to_vec();
        let scatter: Vec<i32> = (0..values.len()).map(|i| indices[if axis == 0 { i / value_shape[1] } else { i % value_shape[1] }] as i32).collect();
        for update in updates() {
            let mut expected = data.clone();
            let mut grad_data = vec![1f32; data.len()];
            let mut grad_values = vec![0f32; values.len()];
            for (i, &value) in values.iter().enumerate() {
                let target = destination(i, value_shape, shape, axis, scatter[i] as usize);
                let original = expected[target];
                expected[target] = fold(original, value, update);
                let (left, right) = match update {
                    IndexingUpdateOp::Assign => (0., 1.),
                    IndexingUpdateOp::Add => (1., 1.),
                    IndexingUpdateOp::Mul => (value, original),
                    IndexingUpdateOp::Min => ((original <= value) as u8 as f32, (value <= original) as u8 as f32),
                    IndexingUpdateOp::Max => ((original >= value) as u8 as f32, (value >= original) as u8 as f32),
                };
                grad_data[target] = left;
                grad_values[i] = right * 2. * expected[target];
            }
            for (gradient, value) in grad_data.iter_mut().zip(expected.iter()) { *gradient *= 2. * value; }
            for select in [false, true] {
                for (track_data, track_values) in [(true, true), (true, false), (false, true)] {
                    let tensor = Tensor::<A<C>, 2>::from_data(TensorData::new(data.clone(), shape), &device).set_require_grad(track_data);
                    let value = Tensor::<A<C>, 2>::from_data(TensorData::new(values.clone(), value_shape), &device).set_require_grad(track_values);
                    let output = if select {
                        tensor.clone().select_assign(axis, Tensor::from_data(indices.map(|i| i as i32), &device), value.clone(), update)
                    } else {
                        tensor.clone().scatter(axis, Tensor::from_data(TensorData::new(scatter.clone(), value_shape), &device), value.clone(), update)
                    };
                    output.to_data().assert_eq(&TensorData::new(expected.clone(), shape), false);
                    let gradients = (output.clone() * output).sum().backward();
                    for (parent, tracked, expected, shape) in [
                        (tensor, track_data, grad_data.clone(), shape),
                        (value, track_values, grad_values.clone(), value_shape),
                    ] {
                        if tracked {
                            parent.grad(&gradients).expect("missing indexing gradient").into_data()
                                .assert_eq(&TensorData::new(expected, shape), false);
                        } else { assert!(parent.grad(&gradients).is_none()); }
                    }
                }
            }
        }
    }
}

#[test]
fn indexing_gradients_cover_all_updates_and_parent_tracking() { gradients::<NoCheckpointing>(); }

#[test]
fn indexing_gradients_preserve_balanced_recomputation() { gradients::<BalancedCheckpointing>(); }

#[test]
fn empty_index_updates_leave_input_unchanged() {
    let device = Default::default();
    for update in updates() {
        let tensor = Tensor::<Inner, 2>::from_data([[2., 5., 3.], [7., 1., 4.]], &device);
        let expected = tensor.to_data();
        tensor.clone().select_assign(1, Tensor::empty([0], &device), Tensor::empty([2, 0], &device), update)
            .into_data().assert_eq(&expected, true);
        tensor.scatter(1, Tensor::empty([2, 0], &device), Tensor::empty([2, 0], &device), update)
            .into_data().assert_eq(&expected, true);
    }
}
