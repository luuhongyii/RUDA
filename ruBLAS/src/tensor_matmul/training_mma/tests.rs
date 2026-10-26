use crate::tensor_matmul::training_gemm::launch_tile;
use half::bf16;
use ruda_core::tensor::{DType, Metadata, Shape, data::TensorData};
use ruda_kernel::tensor::{RudaTensor, transfer::from_data, readback::into_data_sync, permutation::swap_dims};
use ruda_test_runtime::TestRuntime;

type Tensor = RudaTensor<TestRuntime>;

#[test]
fn manual_training_epilogue_swizzle_preserves_coordinates_and_bank_transactions() {
    let index = |row: usize, col: usize| row * 16 + (col ^ ((row / 2 % 2) * 12));
    let mut shared = [usize::MAX; 256];
    for part in 0..2 {
        for r in 0..2 {
            for half_warp in 0..2 {
                let mut banks = [0usize; 32];
                for lane in half_warp * 16..(half_warp + 1) * 16 {
                    let (row, col) = (lane / 4 + r * 8, part * 8 + lane % 4 * 2);
                    let start = index(row, col);
                    assert_eq!(start % 2, 0);
                    for c in 0..2 {
                        assert_eq!(index(row, col + c), start + c);
                        assert_eq!(shared[start + c], usize::MAX);
                        shared[start + c] = row * 16 + col + c;
                        banks[(start + c) % 32] += 1;
                    }
                }
                assert_eq!(banks, [1; 32]);
            }
        }
    }
    for width in [1, 4, 8] {
        let vector = width.min(4);
        let lanes = 32 / vector;
        for t in 0..8 / width {
            for group in 0..width / vector {
                for transaction in 0..32 / lanes {
                    let mut banks = [0usize; 32];
                    for lane in transaction * lanes..(transaction + 1) * lanes {
                        let logical = (lane + t * 32) * width + group * vector;
                        let start = index(logical / 16, logical % 16);
                        assert_eq!(start % vector, 0);
                        for v in 0..vector {
                            assert_eq!(shared[start + v], logical + v);
                            banks[(start + v) % 32] += 1;
                        }
                    }
                    assert_eq!(banks, [1; 32]);
                }
            }
        }
    }
}

fn values(count: usize, seed: usize) -> Vec<bf16> {
    (0..count).map(|i| {
        let bits = ((i * 17 + seed) % 128) | (((i * 7 + seed) % 12 + 119) << 7)
            | (((i * 13 + seed) % 2) << 15);
        bf16::from_bits(bits as u16)
    }).collect()
}

fn output(batch: usize, m: usize, n: usize, dtype: DType, offset: usize) -> (Tensor, Tensor) {
    let count = batch * m * n;
    let shape = Shape::from([count + offset + 8]);
    let data = if dtype == DType::BF16 {
        TensorData::new(vec![bf16::from_f32(-7.0); count + offset + 8], shape)
    } else {
        TensorData::new(vec![-7.0f32; count + offset + 8], shape)
    };
    let storage: Tensor = from_data(data, &Default::default());
    let mut view = storage.clone();
    *view.meta = Metadata::new([batch, m, n], vec![m * n, n, 1]);
    view.handle.offset_start = Some((offset * dtype.size()) as u64);
    (view, storage)
}

fn read(tensor: Tensor) -> Vec<f32> {
    let dtype = tensor.dtype;
    let data = into_data_sync(tensor);
    if dtype == DType::BF16 {
        data.to_vec::<bf16>().unwrap().into_iter().map(bf16::to_f32).collect()
    } else { data.to_vec::<f32>().unwrap() }
}

#[test]
#[ignore = "requires an actual CUDA device and RUDA_REQUIRE_GPU=1"]
fn manual_training_tiles_match_wmma_for_layouts_pipeline_tails_and_offsets() {
    assert!(std::any::type_name::<TestRuntime>().contains("CudaRuntime"));
    assert_eq!(std::env::var("RUDA_REQUIRE_GPU").as_deref(), Ok("1"));
    let mut cases = 0;
    for (bm, bn, bk, asynchronous, warp_m, stages) in [
        (32, 32, 32, false, 32, 1),
        (128, 128, 32, true, 32, 2), (128, 128, 32, true, 32, 3),
        (128, 32, 64, true, 32, 2), (128, 32, 64, true, 32, 3),
        (128, 128, 64, true, 64, 2), (128, 128, 64, true, 64, 3),
        (128, 256, 64, true, 32, 2), (128, 256, 64, true, 32, 3),
    ] {
        for ta in [false, true] {
            for nt in [false, true] {
                for blocks in [1, 2, 4, 7] {
                    let (batch, m, k, n) = (2usize, bm, bk * blocks, bn * 2);
                    let last_k = if blocks == 1 { k } else { k - bk };
                    let pad = if blocks % 2 == 0 { 8 } else { 0 };
                    let (ar, ac) = if ta { (k, m) } else { (m, k) };
                    let (br, bc) = if nt { (n, k) } else { (k, n) };
                    let left = values(batch * ar * (ac + pad), 3);
                    let right = values(br * (bc + pad), 19);
                    let mut a: Tensor = from_data(TensorData::new(left.clone(), [batch, ar, ac + pad]), &Default::default());
                    let mut b: Tensor = from_data(TensorData::new(right.clone(), [1, br, bc + pad]), &Default::default());
                    a.meta.shape_mut()[2] = ac;
                    b.meta.shape_mut()[2] = bc;
                    b.meta.shape_mut()[0] = batch;
                    b.meta.strides_mut()[0] = 0;
                    if ta { a = swap_dims(a, 1, 2); }
                    if nt { b = swap_dims(b, 1, 2); }
                    for dtype in [DType::BF16, DType::F32] {
                        let offset = if blocks == 4 { 3 } else { 0 };
                        let (out, storage) = output(batch, m, n, dtype, offset);
                        let (reference, _) = output(batch, m, n, dtype, 0);
                        let actual = read(launch_tile(a.clone(), b.clone(), out, last_k,
                            bm, bn, bk, asynchronous, warp_m, true, stages).unwrap());
                        let expected = read(launch_tile(a.clone(), b.clone(), reference, last_k,
                            32, 32, 32, false, 32, false, 1).unwrap());
                        assert_eq!(actual.len(), expected.len());
                        let mismatch = actual.iter().zip(&expected).position(|(a, b)| a.to_bits() != b.to_bits());
                        assert_eq!(mismatch, None,
                            "tile={bm}x{bn}x{bk}, stages={stages}, ta={ta}, nt={nt}, blocks={blocks}, dtype={dtype:?}");
                        let whole = read(storage);
                        assert!(whole[..offset].iter().all(|v| *v == -7.0));
                        assert!(whole[offset + actual.len()..].iter().all(|v| *v == -7.0));
                        for z in 0..batch {
                            for (row, col) in [(0, 0), (m / 2, n / 2), (m - 1, n - 1)] {
                                let limit = if z + 1 == batch { last_k } else { k };
                                let sum: f64 = (0..limit).map(|t| {
                                    let ai = z * ar * (ac + pad) + if ta { t * (ac + pad) + row } else { row * (ac + pad) + t };
                                    let bi = if nt { col * (bc + pad) + t } else { t * (bc + pad) + col };
                                    f64::from(left[ai].to_f32()) * f64::from(right[bi].to_f32())
                                }).sum();
                                let tolerance = if dtype == DType::BF16 { 0.008 } else { 2e-5 };
                                let value = f64::from(actual[(z * m + row) * n + col]);
                                assert!(value.is_finite() && (value - sum).abs() <= tolerance * sum.abs().max(1.0));
                            }
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 288);
}
