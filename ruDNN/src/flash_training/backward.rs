//! Recomputed 16x64 Flash backward tiles; no global LxS score/probability buffers.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{cmma::MmaDefinition, ir::MatrixIdent, prelude::*};
use crate::dense_attention::flash_exp2;

#[ruda]
fn dot_tile<F: Float>(a: &Array<F>, b: &Array<F>, tile: &mut SharedMemory<f32>,
    a_row: usize, b_row: usize, offset: usize) {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..2 {
        let mut acc = Array::<Vector<f32, Const<2>>>::new(2usize);
        #[unroll]
        for i in 0usize..2 { acc[i] = Vector::new(0.0f32); }
        #[unroll]
        for chunk in 0usize..8 {
            let mut ar = Array::<Vector<F, Const<2>>>::new(4usize);
            let mut br = Array::<Vector<F, Const<2>>>::new(2usize);
            #[unroll]
            for i in 0usize..4 {
                let mut value = Vector::<F, Const<2>>::empty();
                #[unroll]
                for j in 0usize..2 {
                    let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::A);
                    value[j] = a[(a_row + r as usize) * 128 + chunk * 16 + c as usize];
                }
                ar[i] = value;
            }
            #[unroll]
            for i in 0usize..2 {
                let mut value = Vector::<F, Const<2>>::empty();
                #[unroll]
                for j in 0usize..2 {
                    let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::B);
                    value[j] = b[(b_row + warp * 16 + col * 8 + c as usize) * 128 + chunk * 16 + r as usize];
                }
                br[i] = value;
            }
            let value = def.execute(&ar, &br, &acc);
            #[unroll]
            for i in 0usize..2 { acc[i] = value[i]; }
        }
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::Accumulator);
                tile[offset + r as usize * 64 + warp * 16 + col * 8 + c as usize] = acc[i][j];
            }
        }
    }
}

#[ruda]
fn contract_tile<F: Float>(weights: &SharedMemory<F>, input: &Array<F>, input_row: usize,
    acc: &mut Array<Vector<f32, Const<2>>>) {
    let lane = UNIT_POS as usize % 32;
    let warp = UNIT_POS as usize / 32;
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..4 {
        let mut cv = Array::<Vector<f32, Const<2>>>::new(2usize);
        #[unroll]
        for i in 0usize..2 { cv[i] = acc[col * 2 + i]; }
        #[unroll]
        for chunk in 0usize..4 {
            let mut ar = Array::<Vector<F, Const<2>>>::new(4usize);
            let mut br = Array::<Vector<F, Const<2>>>::new(2usize);
            #[unroll]
            for i in 0usize..4 {
                let mut value = Vector::<F, Const<2>>::empty();
                #[unroll]
                for j in 0usize..2 {
                    let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::A);
                    value[j] = weights[r as usize * 64 + chunk * 16 + c as usize];
                }
                ar[i] = value;
            }
            #[unroll]
            for i in 0usize..2 {
                let mut value = Vector::<F, Const<2>>::empty();
                #[unroll]
                for j in 0usize..2 {
                    let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::B);
                    value[j] = input[(input_row + chunk * 16 + r as usize) * 128 + warp * 32 + col * 8 + c as usize];
                }
                br[i] = value;
            }
            let value = def.execute(&ar, &br, &cv);
            #[unroll]
            for i in 0usize..2 { cv[i] = value[i]; }
        }
        #[unroll]
        for i in 0usize..2 { acc[col * 2 + i] = cv[i]; }
    }
}

#[ruda(launch)]
pub fn contraction<F: Float>(grad: &Array<F>, stats: &Array<f32>, delta: &mut Array<f32>,
    #[define(F)] _dtype: StorageType) {
    let lane = UNIT_POS as usize % 32;
    let row = RUDA_POS_X as usize * 4 + UNIT_POS as usize / 32;
    let mut sum = 0.0f32;
    #[unroll]
    for i in 0usize..4 {
        let at = row * 128 + lane + i * 32;
        sum += f32::cast_from(grad[at]) * stats[delta.len() * 2 + at];
    }
    sum = plane_sum(sum);
    if lane == 0 { delta[row] = sum; }
}

#[ruda(launch)]
pub fn query<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dq: &mut Array<F>, heads: u32, kv_heads: u32,
    scale: f32, #[comptime] causal: bool, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X as usize * 16;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let head = row / 512 % heads as usize;
    let batch = row / (512 * heads as usize);
    let key = (batch * kv_heads as usize + head / (heads / kv_heads) as usize) * 512;
    let mut scores = SharedMemory::<f32>::new(2048usize);
    let mut ds = SharedMemory::<F>::new(1024usize);
    let mut acc = Array::<Vector<f32, Const<2>>>::new(8usize);
    #[unroll]
    for i in 0usize..8 { acc[i] = Vector::new(0.0f32); }
    for block in 0usize..8 {
        dot_tile(q, k, &mut scores, row, key + block * 64, 0usize);
        dot_tile(grad, v, &mut scores, row, key + block * 64, 1024usize);
        sync_ruda();
        #[unroll]
        for i in 0usize..8 {
            let at = thread + i * 128;
            let qr = row + at / 64;
            let mut value = 0.0f32;
            let mut allowed = true;
            if comptime!(causal) { allowed = block * 64 + at % 64 <= qr % 512; }
            if allowed {
                let p = flash_exp2(scores[at] * (scale * 1.4426950408889634f32) - stats[qr * 2] * 1.4426950408889634f32);
                value = p * (scores[1024 + at] - delta[qr]);
            }
            ds[at] = F::cast_from(value);
        }
        sync_ruda();
        contract_tile(&ds, k, key + block * 64, &mut acc);
        sync_ruda();
    }
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..4 {
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::Accumulator);
                dq[(row + r as usize) * 128 + warp * 32 + col * 8 + c as usize] = F::cast_from(acc[col * 2 + i][j] * scale);
            }
        }
    }
}

#[ruda(launch)]
pub fn key_value<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, grad: &Array<F>,
    stats: &Array<f32>, delta: &Array<f32>, dk: &mut Array<F>, dv: &mut Array<F>,
    heads: u32, kv_heads: u32, scale: f32, #[comptime] causal: bool, #[define(F)] _dtype: StorageType) {
    let key = RUDA_POS_X as usize * 16;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let kh = key / 512 % kv_heads as usize;
    let batch = key / (512 * kv_heads as usize);
    let group = (heads / kv_heads) as usize;
    let mut scores = SharedMemory::<f32>::new(2048usize);
    let mut ds = SharedMemory::<F>::new(1024usize);
    let mut probability = SharedMemory::<F>::new(1024usize);
    let mut sum_k = Array::<Vector<f32, Const<2>>>::new(8usize);
    let mut sum_v = Array::<Vector<f32, Const<2>>>::new(8usize);
    #[unroll]
    for i in 0usize..8 { sum_k[i] = Vector::new(0.0f32); sum_v[i] = Vector::new(0.0f32); }
    for head_in_group in 0usize..group {
        let query = (batch * heads as usize + kh * group + head_in_group) * 512;
        let mut acc_k = Array::<Vector<f32, Const<2>>>::new(8usize);
        let mut acc_v = Array::<Vector<f32, Const<2>>>::new(8usize);
        #[unroll]
        for i in 0usize..8 { acc_k[i] = Vector::new(0.0f32); acc_v[i] = Vector::new(0.0f32); }
        for block in 0usize..8 {
            dot_tile(k, q, &mut scores, key, query + block * 64, 0usize);
            dot_tile(v, grad, &mut scores, key, query + block * 64, 1024usize);
            sync_ruda();
            #[unroll]
            for i in 0usize..8 {
                let at = thread + i * 128;
                let qr = query + block * 64 + at % 64;
                let mut p = 0.0f32;
                let mut value = 0.0f32;
                let mut allowed = true;
                if comptime!(causal) { allowed = key % 512 + at / 64 <= qr % 512; }
                if allowed {
                    p = flash_exp2(scores[at] * (scale * 1.4426950408889634f32) - stats[qr * 2] * 1.4426950408889634f32);
                    value = p * (scores[1024 + at] - delta[qr]);
                }
                ds[at] = F::cast_from(value);
                probability[at] = F::cast_from(p);
            }
            sync_ruda();
            contract_tile(&ds, q, query + block * 64, &mut acc_k);
            contract_tile(&probability, grad, query + block * 64, &mut acc_v);
            sync_ruda();
        }
        #[unroll]
        for i in 0usize..8 {
            let mut total_k = sum_k[i];
            let mut total_v = sum_v[i];
            #[unroll]
            for j in 0usize..2 {
                total_k[j] += f32::cast_from(F::cast_from(acc_k[i][j] * scale));
                total_v[j] += f32::cast_from(F::cast_from(acc_v[i][j]));
            }
            sum_k[i] = total_k;
            sum_v[i] = total_v;
        }
    }
    let def = MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    #[unroll]
    for col in 0usize..4 {
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r, c) = def.position_of_nth(lane as u32, (i * 2 + j) as u32, MatrixIdent::Accumulator);
                let at = (key + r as usize) * 128 + warp * 32 + col * 8 + c as usize;
                dk[at] = F::cast_from(sum_k[col * 2 + i][j]);
                dv[at] = F::cast_from(sum_v[col * 2 + i][j]);
            }
        }
    }
}
