use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::{*, barrier::{Barrier, copy_async}}}};

#[cfg(test)]
mod tests;

#[ruda]
fn shared_index(row: usize, chunk: usize, #[comptime] chunks: usize) -> usize {
    row * chunks + (chunk ^ (row % comptime!(chunks.min(8))))
}

#[ruda(launch_unchecked)]
pub(super) fn tiled<F: Float, O: Float, OS: Size>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    out: &mut Array<Vector<O, OS>>, #[comptime] k: usize, #[comptime] n: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] ta: bool, #[comptime] nt: bool,
    #[comptime] m: usize, #[comptime] batch: usize, #[comptime] last_k: usize,
    #[comptime] a_batch: usize, #[comptime] b_batch: usize,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[comptime] warp_m: usize,
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
    let banks = comptime!(if asynchronous { 3usize } else { 1usize });
    let a_stage = comptime!(bm * bk / 8);
    let b_stage = comptime!(bn * bk / 8);
    let mut sa = SharedMemory::<Vector<F, Const<8>>>::new_aligned(a_stage * banks, 32usize);
    let mut sb = SharedMemory::<Vector<F, Const<8>>>::new_aligned(b_stage * banks, 32usize);
    let mut epilogue = SharedMemory::<f32>::new_aligned(bm / warp_m * 512usize, 32usize);
    let mut loaded = Sequence::<Shared<Barrier>>::new();
    #[unroll]
    for _ in 0usize..banks { loaded.push(Barrier::shared(RUDA_DIM, UNIT_POS == 0)); }
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
    for half in 0usize..(bk / 16) {
        #[unroll]
        for i in 0usize..(warp_m / 16) {
            if comptime!(ta) {
                a_rows.push(shared_index(half * 16 + lane % 8 + lane / 16 * 8,
                    (wm + i * 16) / 8 + lane % 16 / 8, bm / 8));
            } else {
                a_rows.push(shared_index(wm + i * 16 + lane % 16, half * 2 + lane / 16, bk / 8));
            }
        }
        #[unroll]
        for j in 0usize..(bn / 16) {
            if comptime!(nt) {
                b_rows.push(shared_index(wn + j * 8 + lane % 8, half * 2 + lane % 16 / 8, bk / 8));
            } else {
                b_rows.push(shared_index(half * 16 + lane % 16, wn / 8 + j, bn / 8));
            }
        }
    }
    let blocks = k / bk - usize::cast_from(RUDA_POS_Y as usize == batch - 1) * ((k - last_k) / bk);
    if comptime!(asynchronous) {
        load_stage(a, b, &mut sa, &mut sb, row, col, 0usize, 0usize, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
        loaded[0].commit_copy_async();
        if blocks > 1 {
            load_stage(a, b, &mut sa, &mut sb, row, col, 1usize, 1usize, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
            loaded[1].commit_copy_async();
        }
    }
    for block in 0usize..blocks {
        let bank = block % banks;
        if comptime!(asynchronous) {
            if bank == 0 { loaded[0].arrive_and_wait(); }
            else if bank == 1 { loaded[1].arrive_and_wait(); }
            else { loaded[2].arrive_and_wait(); }
            if block + 2 < blocks {
                let next_bank = (bank + 2) % banks;
                load_stage(a, b, &mut sa, &mut sb, row, col, block + 2, next_bank, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
                if next_bank == 0 { loaded[0].commit_copy_async(); }
                else if next_bank == 1 { loaded[1].commit_copy_async(); }
                else { loaded[2].commit_copy_async(); }
            }
        } else {
            load_stage(a, b, &mut sa, &mut sb, row, col, block, bank, lda, ldb, ta, nt, a_batch, b_batch, bm, bn, bk, asynchronous, warp_m);
            sync_ruda();
        }
        #[unroll]
        for half in 0usize..(bk / 16) {
            let mut ar = Sequence::<Vector<F, Const<8>>>::new();
            let mut br = Sequence::<Vector<F, Const<4>>>::new();
            #[unroll]
            for i in 0usize..(warp_m / 16) {
                let offset = bank * a_stage + a_rows[half * (warp_m / 16) + i];
                ar.push(def.load_matrix_vector::<_, Const<8>>(&sa.slice(offset, offset + 1), 4usize, ta));
            }
            #[unroll]
            for j in 0usize..(bn / 16) {
                let offset = bank * b_stage + b_rows[half * (bn / 16) + j];
                br.push(def.load_matrix_vector::<_, Const<4>>(&sb.slice(offset, offset + 1), 2usize, !nt));
            }
            #[unroll]
            for i in 0usize..(warp_m / 16) {
                #[unroll]
                for j in 0usize..(bn / 16) {
                    def.execute_vectors_inplace(&ar[i], &br[j], acc.index_mut(i * (bn / 16) + j));
                }
            }
        }
        if comptime!(!asynchronous) { sync_ruda(); }
    }
    #[unroll]
    for i in 0usize..(warp_m / 16) {
        #[unroll]
        for j in 0usize..(bn / 32) {
            #[unroll]
            for part in 0usize..2 {
                #[unroll]
                for r in 0usize..2 {
                    let value = acc[i * (bn / 16) + j * 2 + part];
                    #[unroll]
                    for c in 0usize..2 {
                        epilogue[warp * 256 + (lane / 4 + r * 8) * 16 + part * 8 + lane % 4 * 2 + c] = value[r * 2 + c];
                    }
                }
            }
            sync_plane();
            #[unroll]
            for t in 0usize..(8 / output_width) {
                let pos = (lane + t * 32) * output_width;
                let mut values = Vector::<O, OS>::empty();
                #[unroll]
                for v in 0usize..output_width { values[v] = O::cast_from(epilogue[warp * 256 + pos + v]); }
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
    #[comptime] warp_m: usize) {
    let threads = comptime!(bm / warp_m * 64);
    #[unroll]
    for i in 0usize..(bm * bk / (threads * 8)) {
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
    #[unroll]
    for i in 0usize..(bn * bk / (threads * 8)) {
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
