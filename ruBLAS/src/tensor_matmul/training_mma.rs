use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::{*, barrier::{copy_async, commit_copy_group, wait_copy_group}}}};

#[cfg(test)]
mod tests;

#[ruda]
fn shared_index(row: usize, chunk: usize, #[comptime] chunks: usize) -> usize {
    row * chunks + (chunk ^ ((row / comptime!(8usize.div_ceil(chunks))) % comptime!(chunks.min(8))))
}

#[ruda]
fn epilogue_index(row: usize, col: usize) -> usize {
    row * 16 + (col ^ ((row / 2 % 2) * 12))
}

#[ruda(launch_unchecked)]
pub(super) fn tiled_compact<F: Float, O: Float, OS: Size>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    out: &mut Array<Vector<O, OS>>, #[comptime] k: usize, #[comptime] n: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] ta: bool, #[comptime] nt: bool,
    #[comptime] m: usize, #[comptime] batch: usize, #[comptime] last_k: usize,
    #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] warp_m: usize, #[comptime] stages: usize,
    #[define(F)] _input: StorageType, #[define(O)] _output: StorageType,
    #[define(OS)] _output_vector_size: usize) {
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let output_width = out.vector_size().comptime();
    let row = RUDA_POS_X as usize % (m / bm) * bm;
    let col = RUDA_POS_X as usize / (m / bm) * bn;
    let wm = warp / 2 * warp_m;
    let wn = warp % 2 * (bn / 2);
    let banks = comptime!(if asynchronous { stages } else { 1usize });
    let a_stage = comptime!(bm * bk / 8);
    let b_stage = comptime!(bn * bk / 8);
    let mut sa = SharedMemory::<Vector<F, Const<8>>>::new_aligned(a_stage * banks, 32usize);
    let mut sb = SharedMemory::<Vector<F, Const<8>>>::new_aligned(b_stage * banks, 32usize);
    let mut epilogue = SharedMemory::<f32>::new_aligned(bm / warp_m * 512usize, 32usize);
    let def = cmma::MmaDefinition::<F, F, f32>::new(16usize, 8usize, 16usize);
    let mut acc = Sequence::<Vector<f32, Const<4>>>::new();
    #[unroll]
    for _ in 0usize..(warp_m / 16 * (bn / 16)) {
        let mut fragment = Vector::<f32, Const<4>>::empty();
        #[unroll]
        for i in 0usize..4 { fragment[i] = 0.0f32; }
        acc.push(fragment);
    }
    let mut a_rows = Sequence::<usize>::new();
    let mut b_rows = Sequence::<usize>::new();
    #[unroll]
    for i in 0usize..(warp_m / 16) {
        if comptime!(ta) {
            a_rows.push(shared_index(lane % 8 + lane / 16 * 8,
                (wm + i * 16) / 8 + lane % 16 / 8, bm / 8));
        } else {
            a_rows.push(shared_index(wm + i * 16 + lane % 16, lane / 16, bk / 8));
        }
    }
    #[unroll]
    for j in 0usize..(bn / 16) {
        if comptime!(nt) {
            b_rows.push(shared_index(wn + j * 8 + lane % 8, lane % 16 / 8, bk / 8));
        } else {
            b_rows.push(shared_index(lane % 16, wn / 8 + j, bn / 8));
        }
    }
    let blocks = k / bk - usize::cast_from(RUDA_POS_Y as usize == batch - 1) * ((k - last_k) / bk);
    if comptime!(asynchronous) {
        #[unroll]
        for stage in 0usize..(banks - 1) {
            if stage < blocks {
                load_stage(a, b, &mut sa, &mut sb, row, col, stage, stage, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m, 0usize, 1usize);
            }
            commit_copy_group();
        }
    }
    for block in 0usize..blocks {
        let bank = block % banks;
        if comptime!(asynchronous) {
            wait_copy_group(comptime!((banks - 2) as u32));
            sync_ruda();
        } else {
            load_stage(a, b, &mut sa, &mut sb, row, col, block, bank, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m, 0usize, 1usize);
            sync_ruda();
        }
        #[unroll]
        for half in 0usize..(bk / 16) {
            if comptime!(asynchronous) {
                if block + banks - 1 < blocks {
                    let next_bank = (bank + banks - 1) % banks;
                    load_stage(a, b, &mut sa, &mut sb, row, col, block + banks - 1, next_bank, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m, half, bk / 16);
                }
            }
            let mut ar = Sequence::<Vector<F, Const<8>>>::new();
            #[unroll]
            for i in 0usize..(warp_m / 16) {
                let base = bank * a_stage + a_rows[i];
                let mut offset = base;
                if comptime!(ta) {
                    offset = base + half * 16 * (bm / 8);
                } else if comptime!(bk.is_power_of_two()) {
                    offset = base ^ (half * 2);
                } else {
                    offset = bank * a_stage + shared_index(wm + i * 16 + lane % 16, half * 2 + lane / 16, bk / 8);
                }
                ar.push(def.load_matrix_vector::<_, Const<8>>(&sa.slice(offset, offset + 1), 4usize, ta));
            }
            #[unroll]
            for j in 0usize..(bn / 16) {
                let base = bank * b_stage + b_rows[j];
                let mut offset = base;
                if comptime!(!nt) {
                    offset = base + half * 16 * (bn / 8);
                } else if comptime!(bk.is_power_of_two()) {
                    offset = base ^ (half * 2);
                } else {
                    offset = bank * b_stage + shared_index(wn + j * 8 + lane % 8, half * 2 + lane % 16 / 8, bk / 8);
                }
                let br = def.load_matrix_vector::<_, Const<4>>(&sb.slice(offset, offset + 1), 2usize, !nt);
                #[unroll]
                for i in 0usize..(warp_m / 16) {
                    def.execute_vectors_inplace(&ar[i], &br, acc.index_mut(i * (bn / 16) + j));
                }
            }
        }
        if comptime!(asynchronous) { commit_copy_group(); }
        if comptime!(!asynchronous) { sync_ruda(); }
    }
    if comptime!(asynchronous) {
        wait_copy_group(0u32);
        sync_ruda();
    }
    let mut pairs = epilogue.to_slice_mut().into_vectorized().with_vector_size::<Const<2>>();
    let quads = epilogue.to_slice().into_vectorized().with_vector_size::<Const<4>>();
    #[unroll]
    for i in 0usize..(warp_m / 16) {
        #[unroll]
        for j in 0usize..(bn / 32) {
            #[unroll]
            for part in 0usize..2 {
                #[unroll]
                for r in 0usize..2 {
                    let value = acc[i * (bn / 16) + j * 2 + part];
                    let mut pair = Vector::<f32, Const<2>>::empty();
                    #[unroll]
                    for c in 0usize..2 { pair[c] = value[r * 2 + c]; }
                    pairs[(warp * 256 + epilogue_index(lane / 4 + r * 8, part * 8 + lane % 4 * 2)) / 2] = pair;
                }
            }
            sync_plane();
            #[unroll]
            for t in 0usize..(8 / output_width) {
                let pos = (lane + t * 32) * output_width;
                let mut values = Vector::<O, OS>::empty();
                if comptime!(output_width >= 4) {
                    #[unroll]
                    for group in 0usize..(output_width / 4) {
                        let value = quads[(warp * 256 + epilogue_index(pos / 16, pos % 16 + group * 4)) / 4];
                        #[unroll]
                        for v in 0usize..4 { values[group * 4 + v] = O::cast_from(value[v]); }
                    }
                } else {
                    #[unroll]
                    for v in 0usize..output_width {
                        values[v] = O::cast_from(epilogue[warp * 256 + epilogue_index(pos / 16, pos % 16 + v)]);
                    }
                }
                out[((RUDA_POS_Y as usize * m + row + wm + i * 16 + pos / 16) * n + col + wn + j * 16 + pos % 16) / output_width] = values;
            }
            sync_plane();
        }
    }
}

#[ruda]
fn load_stage<F: Float>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    sa: &mut SharedMemory<Vector<F, Const<8>>>, sb: &mut SharedMemory<Vector<F, Const<8>>>,
    row: usize, col: usize, block: usize, bank: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] ta: bool, #[comptime] nt: bool,
    #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] warp_m: usize, #[comptime] copy_group: usize, #[comptime] copy_groups: usize) {
    let threads = comptime!(bm / warp_m * 64);
    let a_copies = comptime!(bm * bk / (threads * 8));
    let a_per_group = comptime!(a_copies.div_ceil(copy_groups));
    #[unroll]
    for i in comptime!((copy_group * a_per_group).min(a_copies))..comptime!(((copy_group + 1) * a_per_group).min(a_copies)) {
        let p = UNIT_POS as usize + i * threads;
        if comptime!(ta) {
            let source = (RUDA_POS_Y as usize * a_batch + (block * bk + p / (bm / 8)) * lda + row) / 8 + p % (bm / 8);
            let target = bank * (bm * bk / 8) + shared_index(p / (bm / 8), p % (bm / 8), bm / 8);
            if comptime!(asynchronous) { copy_async(&a.slice(source, source + 1), &mut sa.slice_mut(target, target + 1), 8u32); }
            else { sa[target] = a[source]; }
        } else {
            let source = (RUDA_POS_Y as usize * a_batch + (row + p / (bk / 8)) * lda + block * bk) / 8 + p % (bk / 8);
            let target = bank * (bm * bk / 8) + shared_index(p / (bk / 8), p % (bk / 8), bk / 8);
            if comptime!(asynchronous) { copy_async(&a.slice(source, source + 1), &mut sa.slice_mut(target, target + 1), 8u32); }
            else { sa[target] = a[source]; }
        }
    }
    let b_copies = comptime!(bn * bk / (threads * 8));
    let b_per_group = comptime!(b_copies.div_ceil(copy_groups));
    #[unroll]
    for i in comptime!((copy_group * b_per_group).min(b_copies))..comptime!(((copy_group + 1) * b_per_group).min(b_copies)) {
        let p = UNIT_POS as usize + i * threads;
        if comptime!(nt) {
            let source = (RUDA_POS_Y as usize * b_batch + (col + p / (bk / 8)) * ldb + block * bk) / 8 + p % (bk / 8);
            let target = bank * (bn * bk / 8) + shared_index(p / (bk / 8), p % (bk / 8), bk / 8);
            if comptime!(asynchronous) { copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 8u32); }
            else { sb[target] = b[source]; }
        } else {
            let source = (RUDA_POS_Y as usize * b_batch + (block * bk + p / (bn / 8)) * ldb + col) / 8 + p % (bn / 8);
            let target = bank * (bn * bk / 8) + shared_index(p / (bn / 8), p % (bn / 8), bn / 8);
            if comptime!(asynchronous) { copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 8u32); }
            else { sb[target] = b[source]; }
        }
    }
}
