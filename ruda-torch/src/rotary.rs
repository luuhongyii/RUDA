//! HF half-rotation with the original low-precision multiplication boundaries.
use super::*;
use super::scientific::address;

#[ruda]
fn value<S: Float + RudaElement>(
    x: &Tensor<S>, cos: &Tensor<S>, sin: &Tensor<S>, pos: usize,
    #[comptime] backward: bool,
) -> S {
    let width = x.shape(3);
    let half = width / 2;
    let col = pos % width;
    let token = pos / width % x.shape(2);
    let batch = pos / width / x.shape(2) / x.shape(1);
    let mut partner = pos + half;
    if col >= half { partner = pos - half; }
    let ci = (batch % cos.shape(0)) * cos.stride(0)
        + (token % cos.shape(2)) * cos.stride(2) + col * cos.stride(3);
    let mut si = (batch % sin.shape(0)) * sin.stride(0)
        + (token % sin.shape(2)) * sin.stride(2) + col * sin.stride(3);
    let left = S::cast_from(f32::cast_from(x[address(x, pos)]) * f32::cast_from(cos[ci]));
    let mut other = f32::cast_from(x[address(x, partner)]);
    if comptime!(backward) {
        let paired_col = partner % width;
        si = (batch % sin.shape(0)) * sin.stride(0)
            + (token % sin.shape(2)) * sin.stride(2) + paired_col * sin.stride(3);
        if col >= half { other = -other; }
    } else {
        if col < half { other = -other; }
    }
    let right = S::cast_from(other * f32::cast_from(sin[si]));
    S::cast_from(f32::cast_from(left) + f32::cast_from(right))
}

#[ruda(launch)]
fn qk<S: Float + RudaElement>(
    q: &Tensor<S>, k: &Tensor<S>, cos: &Tensor<S>, sin: &Tensor<S>,
    out_q: &mut Tensor<S>, out_k: &mut Tensor<S>,
    #[comptime] backward: bool, #[comptime] need_q: bool, #[comptime] need_k: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    let mut q_len = 0usize;
    if comptime!(need_q) { q_len = q.len(); }
    if pos < q_len {
        out_q[pos] = value(q, cos, sin, pos, backward);
    } else {
        if comptime!(need_k) {
            let index = pos - q_len;
            if index < k.len() { out_k[index] = value(k, cos, sin, index, backward); }
        }
    }
}

#[ruda]
fn pair<S: Float + RudaElement, V: Size>(
    x: &Tensor<Vector<S, V>>, cos: &Tensor<Vector<S, V>>, sin: &Tensor<Vector<S, V>>,
    out: &mut Tensor<Vector<S, V>>, pos: usize, #[comptime] backward: bool,
) {
    let vector = x.vector_size().comptime();
    let width = x.shape(3);
    let half = width / 2usize;
    let row = pos / (half / vector);
    let col = pos % (half / vector) * vector;
    let logical = row * (width / vector) + col / vector;
    let source = super::pointwise::vector_offset(x, logical, false);
    let first = x[source];
    let second = x[source + half / vector];
    let token = row % x.shape(2);
    let batch = row / x.shape(2) / x.shape(1);
    let ci = ((batch % cos.shape(0)) * cos.stride(0) + (token % cos.shape(2)) * cos.stride(2) + col) / vector;
    let si = ((batch % sin.shape(0)) * sin.stride(0) + (token % sin.shape(2)) * sin.stride(2) + col) / vector;
    let cos_first = cos[ci];
    let cos_second = cos[ci + half / vector];
    let sin_first = sin[si];
    let sin_second = sin[si + half / vector];
    let mut out_first = Vector::<S, V>::empty();
    let mut out_second = Vector::<S, V>::empty();
    #[unroll] for lane in 0usize..vector {
        let a = f32::cast_from(first[lane]);
        let b = f32::cast_from(second[lane]);
        let left = S::cast_from(a * f32::cast_from(cos_first[lane]));
        let right = S::cast_from(b * f32::cast_from(cos_second[lane]));
        let mut cross_first = S::cast_from(0.0f32);
        let mut cross_second = S::cast_from(0.0f32);
        if comptime!(backward) {
            cross_first = S::cast_from(b * f32::cast_from(sin_second[lane]));
            cross_second = S::cast_from(-a * f32::cast_from(sin_first[lane]));
        } else {
            cross_first = S::cast_from(-b * f32::cast_from(sin_first[lane]));
            cross_second = S::cast_from(a * f32::cast_from(sin_second[lane]));
        }
        out_first[lane] = S::cast_from(f32::cast_from(left) + f32::cast_from(cross_first));
        out_second[lane] = S::cast_from(f32::cast_from(right) + f32::cast_from(cross_second));
    }
    out[logical] = out_first;
    out[logical + half / vector] = out_second;
}

#[ruda(launch)]
fn qk_pairs<S: Float + RudaElement, V: Size>(
    q: &Tensor<Vector<S, V>>, k: &Tensor<Vector<S, V>>,
    cos: &Tensor<Vector<S, V>>, sin: &Tensor<Vector<S, V>>,
    out_q: &mut Tensor<Vector<S, V>>, out_k: &mut Tensor<Vector<S, V>>,
    #[comptime] backward: bool, #[comptime] need_q: bool, #[comptime] need_k: bool,
    #[define(V)] _vector_size: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    let mut q_pairs = 0usize;
    if comptime!(need_q) { q_pairs = q.len() / 2usize; }
    if pos < q_pairs { pair(q, cos, sin, out_q, pos, backward); }
    else if comptime!(need_k) {
        let index = pos - q_pairs;
        if index < k.len() / 2usize { pair(k, cos, sin, out_k, index, backward); }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 6);
    assert_eq!(ints.len(), if op == 652 { 0 } else { 2 });
    let q = &xs[0]; let k = &xs[1];
    assert_eq!(q.shape.len(), 4); assert_eq!(k.shape.len(), 4);
    assert!(q.dtype <= 2 && xs.iter().all(|x| x.dtype == q.dtype));
    assert!(q.shape[3] > 0 && q.shape[3] % 2 == 0);
    assert_eq!((k.shape[0], k.shape[2], k.shape[3]), (q.shape[0], q.shape[2], q.shape[3]));
    for coefficient in [&xs[2], &xs[3]] {
        assert_eq!(coefficient.shape.len(), 4);
        assert_eq!(coefficient.shape[1], 1);
        assert!(coefficient.shape[0] == 1 || coefficient.shape[0] == q.shape[0]);
        assert!(coefficient.shape[2] == 1 || coefficient.shape[2] == q.shape[2]);
        assert_eq!(coefficient.shape[3], q.shape[3]);
    }
    for (input, output) in [(q, &xs[4]), (k, &xs[5])] {
        assert_eq!(input.shape, output.shape);
    }
    let need_q = op == 652 || ints[0] == 1;
    let need_k = op == 652 || ints[1] == 1;
    if op == 653 { assert!(ints.iter().all(|&x| x == 0 || x == 1)); }
    for (output, needed) in [(&xs[4], need_q), (&xs[5], need_k)] {
        if needed {
            let mut stride = 1usize;
            for (&dim, &actual) in output.shape.iter().zip(&output.strides).rev() {
                assert!(dim <= 1 || actual == stride, "rotary output must be contiguous");
                stride = stride.checked_mul(dim).expect("rotary shape overflow");
            }
        }
    }
    let work = (if need_q {q.len} else {0usize}).checked_add(if need_k {k.len} else {0usize})
        .filter(|&n| n <= u32::MAX as usize).expect("rotary grid overflow");
    if work == 0 { return; }
    let vector = [if q.dtype == 0 { 4 } else { 8 }, 4].into_iter().find(|&size| q.shape[3] % (2 * size) == 0 && xs.iter().enumerate()
        .filter(|(index, _)| match *index { 0 | 4 => need_q, 1 | 5 => need_k, _ => true })
        .all(|(_, v)| v.strides.last() == Some(&1)
            && v.shape.iter().zip(&v.strides).take(v.shape.len() - 1).all(|(&dim, &stride)| dim <= 1 || stride % size == 0)
            && v.handle.offset_start.unwrap_or(0) % (size * super::element_bytes(v.dtype)) as u64 == 0)).unwrap_or(1);
    let c = client(); let grid = RudaCount::Static((work / if vector > 1 { 2 * vector } else { 1 }).div_ceil(128) as u32, 1, 1);
    macro_rules! run { ($s:ty) => { unsafe {
        if vector > 1 { qk_pairs::launch::<$s,CudaRuntime>(&c,grid,RudaDim::new_1d(128),
            q.arg(),k.arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),op == 653,need_q,need_k,vector); }
        else {
        qk::launch::<$s,CudaRuntime>(&c,grid,RudaDim::new_1d(128),
            q.arg(),k.arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),op == 653,need_q,need_k);
        }
    } }; }
    match q.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
