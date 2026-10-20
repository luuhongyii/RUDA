use ruda_tensor::{BoolStore, DType, TensorData, api::{Bool, Tensor}};
use alloc::{vec, vec::Vec};

type B = super::CudaNative;

#[test]
fn bool_creation_resolves_backend_storage() {
    let device = Default::default();
    let expected = TensorData::from([[true, false, true], [false, true, false]]);
    let default_dtype = Tensor::<B, 2, Bool>::from_data(expected.clone(), &device).dtype();
    for storage in [BoolStore::Native, BoolStore::U8, BoolStore::U32] {
        let dtype = DType::Bool(storage);
        let input = Tensor::<B, 2, Bool>::from_data(
            expected.clone().convert_dtype(dtype), (&device, dtype),
        );
        assert_eq!(input.dtype(), default_dtype);
        input.into_data().assert_eq(&expected, false);
    }
}

#[test]
fn mask_broadcast_expands_beyond_both_values_and_one_block() {
    let device = Default::default();
    for rows in [2, 1025] {
        let mask: Vec<bool> = (0..rows * 3).map(|i| i % 3 == i / 3 % 3).collect();
        let expected: Vec<f32> = mask.iter().enumerate().map(|(i, &selected)| {
            let column = i % 3;
            if selected { [10., 20., 30.][column] } else { [1., 2., 3.][column] }
        }).collect();
        for shared in [false, true] {
            let input = Tensor::<B, 2>::from_data([[1., 2., 3.]], &device);
            let value = Tensor::<B, 2>::from_data([[10., 20., 30.]], &device);
            let retained = shared.then(|| (input.clone(), value.clone()));
            let condition = Tensor::<B, 2, Bool>::from_data(
                TensorData::new(mask.clone(), [rows, 3]), &device,
            );
            let output = input.mask_where(condition, value);
            assert_eq!(output.dims(), [rows, 3]);
            output.into_data().assert_eq(&TensorData::new(expected.clone(), [rows, 3]), true);
            if let Some((input, value)) = retained {
                input.into_data().assert_eq(&TensorData::from([[1f32, 2., 3.]]), true);
                value.into_data().assert_eq(&TensorData::from([[10f32, 20., 30.]]), true);
            }
        }
    }
}

#[test]
fn one_sided_reduction_and_permuted_broadcast() {
    let device = Default::default();
    let left = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device);
    let right = Tensor::<B, 2>::from_data([[10., 20., 30.]], &device);
    let output = Tensor::<B, 2>::einsum("ij,kl->il", [left.into(), right.into()]);
    output.into_data().assert_eq(&TensorData::from([[30f32, 60., 90.], [70., 140., 210.]]), true);
    let left = Tensor::<B, 4>::from_data(TensorData::new(vec![3., 7.], [2, 1, 1, 1]), &device);
    let right = Tensor::<B, 4>::from_data(TensorData::new(vec![10., 20., 30.], [1, 3, 1, 1]), &device);
    (left * right).into_data().assert_eq(
        &TensorData::new(vec![30f32, 60., 90., 70., 140., 210.], [2, 3, 1, 1]), true,
    );
}

#[test]
fn pitched_singleton_reshape_preserves_values() {
    let device = Default::default();
    let row = Tensor::<B, 2>::from_data([[10., 20., 30.]], &device);
    row.clone().transpose().reshape([1, 3, 1, 1]).into_data().assert_eq(
        &TensorData::new(vec![10f32, 20., 30.], [1, 3, 1, 1]), true,
    );
    row.reshape([3, 1, 1]).into_data().assert_eq(
        &TensorData::new(vec![10f32, 20., 30.], [3, 1, 1]), true,
    );
    let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::<B, 5>::from_data(TensorData::new(values.clone(), [1, 6, 1, 4, 1]), &device);
    tensor.reshape([2, 3, 1, 2, 2, 1]).into_data().assert_eq(
        &TensorData::new(values, [2, 3, 1, 2, 2, 1]), true,
    );
}

macro_rules! empty_broadcast_cases {
    ($backend:ty) => {
        use ruda_tensor::{TensorData, api::{Bool, Int, Tensor}};
        use alloc::{vec, vec::Vec};
        type B = $backend;

        fn shapes() -> Vec<([usize; 3], [usize; 3], [usize; 3])> {
            let pairs = [
                ([0, 1, 1], [1, 1, 1], [0, 1, 1]),
                ([2, 0, 3], [1, 1, 3], [2, 0, 3]),
                ([2, 0, 3], [1, 1, 1], [2, 0, 3]),
                ([0, 1, 3], [1, 2, 1], [0, 2, 3]),
            ];
            pairs.into_iter().flat_map(|(lhs, rhs, out)| [(lhs, rhs, out), (rhs, lhs, out)]).collect()
        }

        #[test]
        fn empty_float_broadcast_owned_and_shared() {
            let device = Default::default();
            for (lhs, rhs, expected) in shapes() {
                for shared in [false, true] {
                    for operation in 0..4 {
                        let a_data = TensorData::new(vec![2f32; lhs.iter().product()], lhs);
                        let b_data = TensorData::new(vec![3f32; rhs.iter().product()], rhs);
                        let a = Tensor::<B, 3>::from_data(a_data.clone(), &device);
                        let b = Tensor::<B, 3>::from_data(b_data.clone(), &device);
                        let retained = shared.then(|| (a.clone(), b.clone()));
                        let out = match operation {
                            0 => (a * b).into_data(),
                            1 => a.powf(b).into_data(),
                            2 => a.powi(b).into_data(),
                            _ => a.equal(b).float().into_data(),
                        };
                        out.assert_eq(&TensorData::new(Vec::<f32>::new(), expected), true);
                        if let Some((a, b)) = retained {
                            a.into_data().assert_eq(&a_data, true);
                            b.into_data().assert_eq(&b_data, true);
                        }
                    }
                }
            }
        }

        #[test]
        fn empty_integer_broadcast_owned_and_shared() {
            let device = Default::default();
            for (lhs, rhs, expected) in shapes() {
                for shared in [false, true] {
                    for operation in 0..3 {
                        let a_data = TensorData::new(vec![2i32; lhs.iter().product()], lhs);
                        let b_data = TensorData::new(vec![3i32; rhs.iter().product()], rhs);
                        let a = Tensor::<B, 3, Int>::from_data(a_data.clone(), &device);
                        let b = Tensor::<B, 3, Int>::from_data(b_data.clone(), &device);
                        let retained = shared.then(|| (a.clone(), b.clone()));
                        let out = match operation {
                            0 => a * b,
                            1 => a.bitwise_and(b),
                            _ => a.powi(b),
                        };
                        out.into_data().assert_eq(&TensorData::new(Vec::<i32>::new(), expected), true);
                        if let Some((a, b)) = retained {
                            a.into_data().assert_eq(&a_data, true);
                            b.into_data().assert_eq(&b_data, true);
                        }
                    }
                }
            }
        }

        #[test]
        fn empty_bool_broadcast_owned_and_shared() {
            let device = Default::default();
            for (lhs, rhs, expected) in shapes() {
                for shared in [false, true] {
                    for use_or in [false, true] {
                        let a_data = TensorData::new(vec![true; lhs.iter().product()], lhs);
                        let b_data = TensorData::new(vec![false; rhs.iter().product()], rhs);
                        let a = Tensor::<B, 3, Bool>::from_data(a_data.clone(), &device);
                        let b = Tensor::<B, 3, Bool>::from_data(b_data.clone(), &device);
                        let retained = shared.then(|| (a.clone(), b.clone()));
                        let out = if use_or { a.bool_or(b) } else { a.bool_and(b) };
                        assert_eq!(out.dims(), expected);
                        assert!(out.into_data().try_into_vec_as::<bool>().unwrap().is_empty());
                        if let Some((a, b)) = retained {
                            a.into_data().assert_eq(&a_data, false);
                            b.into_data().assert_eq(&b_data, false);
                        }
                    }
                }
            }
        }

        #[test]
        fn empty_mask_broadcast_owned_and_shared() {
            let device = Default::default();
            for empty_inputs in 1..8 {
                for shared in [false, true] {
                    let shape = |bit| if empty_inputs & bit != 0 { [2, 0, 3] } else { [1, 1, 3] };
                    let lhs = shape(1);
                    let rhs = shape(2);
                    let mask_shape = shape(4);
                    let a_data = TensorData::new(vec![2f32; lhs.iter().product()], lhs);
                    let b_data = TensorData::new(vec![3f32; rhs.iter().product()], rhs);
                    let m_data = TensorData::new(vec![true; mask_shape.iter().product()], mask_shape);
                    let a = Tensor::<B, 3>::from_data(a_data.clone(), &device);
                    let b = Tensor::<B, 3>::from_data(b_data.clone(), &device);
                    let mask = Tensor::<B, 3, Bool>::from_data(m_data.clone(), &device);
                    let retained = shared.then(|| (a.clone(), b.clone(), mask.clone()));
                    a.mask_where(mask, b).into_data().assert_eq(
                        &TensorData::new(Vec::<f32>::new(), [2, 0, 3]), true,
                    );
                    if let Some((a, b, mask)) = retained {
                        a.into_data().assert_eq(&a_data, true);
                        b.into_data().assert_eq(&b_data, true);
                        mask.into_data().assert_eq(&m_data, false);
                    }
                }
            }
        }
    };
}

mod empty_native {
    empty_broadcast_cases!(super::super::CudaNative);
}

#[cfg(feature = "cuda-fusion")]
mod empty_fusion {
    empty_broadcast_cases!(super::super::Cuda);
}
