use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn sigmoid<F: Float>(x: F) -> F {
    F::new(1.0) / (F::new(1.0) + (-x).exp())
}

#[ruda]
fn gate<S: Float, F: Float>(
    input: &Tensor<S>, hidden: &Tensor<S>, bias_i: &Tensor<S>, bias_h: &Tensor<S>,
    offset: usize, bias_offset: usize, #[comptime] has_bias: bool,
) -> F {
    let mut bi = F::new(0.0);
    let mut bh = F::new(0.0);
    if comptime!(has_bias) {
        bi = F::cast_from(bias_i[address(bias_i, bias_offset)]);
        bh = F::cast_from(bias_h[address(bias_h, bias_offset)]);
    }
    F::cast_from(input[address(input, offset)]) + F::cast_from(hidden[address(hidden, offset)]) + bi + bh
}

#[ruda(launch)]
fn lstm_forward<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, hidden: &Tensor<S>, cx: &Tensor<S>,
    bias_i: &Tensor<S>, bias_h: &Tensor<S>, hy: &mut Tensor<S>,
    cy: &mut Tensor<S>, workspace: &mut Tensor<S>, #[comptime] has_bias: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < cx.len() {
        let width = cx.shape(1);
        let column = p % width;
        let base = (p / width) * 4usize * width + column;
        let i = sigmoid::<F>(gate::<S, F>(input, hidden, bias_i, bias_h, base, column, has_bias));
        let f = sigmoid::<F>(gate::<S, F>(input, hidden, bias_i, bias_h, base + width, column + width, has_bias));
        let c = gate::<S, F>(input, hidden, bias_i, bias_h, base + 2usize * width, column + 2usize * width, has_bias).tanh();
        let o = sigmoid::<F>(gate::<S, F>(input, hidden, bias_i, bias_h, base + 3usize * width, column + 3usize * width, has_bias));
        let next_c = f * F::cast_from(cx[address(cx, p)]) + i * c;
        let next_h = o * next_c.tanh();
        hy[address(hy, p)] = S::cast_from(next_h);
        cy[address(cy, p)] = S::cast_from(next_c);
        workspace[address(workspace, base)] = S::cast_from(i);
        workspace[address(workspace, base + width)] = S::cast_from(f);
        workspace[address(workspace, base + 2usize * width)] = S::cast_from(c);
        workspace[address(workspace, base + 3usize * width)] = S::cast_from(o);
    }
}

#[ruda(launch)]
fn lstm_backward<S: Float + RudaElement, F: Float + RudaElement>(
    grad_hy: &Tensor<S>, grad_cy: &Tensor<S>, cx: &Tensor<S>, cy: &Tensor<S>,
    workspace: &Tensor<S>, grad_gates: &mut Tensor<S>, grad_cx: &mut Tensor<S>,
    #[comptime] has_hy: bool, #[comptime] has_cy: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < cx.len() {
        let width = cx.shape(1);
        let base = (p / width) * 4usize * width + p % width;
        let i = F::cast_from(workspace[address(workspace, base)]);
        let f = F::cast_from(workspace[address(workspace, base + width)]);
        let c = F::cast_from(workspace[address(workspace, base + 2usize * width)]);
        let o = F::cast_from(workspace[address(workspace, base + 3usize * width)]);
        let mut gh = F::new(0.0);
        let mut gc = F::new(0.0);
        if comptime!(has_hy) { gh = F::cast_from(grad_hy[address(grad_hy, p)]); }
        if comptime!(has_cy) { gc = F::cast_from(grad_cy[address(grad_cy, p)]); }
        let one = F::new(1.0);
        let tanh_cy = F::cast_from(cy[address(cy, p)]).tanh();
        let go = gh * tanh_cy;
        let dc = gh * o * (one - tanh_cy * tanh_cy) + gc;
        let gi = dc * c;
        let gf = dc * F::cast_from(cx[address(cx, p)]);
        let gn = dc * i;
        grad_gates[address(grad_gates, base)] = S::cast_from(gi * (one - i) * i);
        grad_gates[address(grad_gates, base + width)] = S::cast_from(gf * (one - f) * f);
        grad_gates[address(grad_gates, base + 2usize * width)] = S::cast_from(gn * (one - c * c));
        grad_gates[address(grad_gates, base + 3usize * width)] = S::cast_from(go * (one - o) * o);
        grad_cx[address(grad_cx, p)] = S::cast_from(dc * f);
    }
}

#[ruda(launch)]
fn gru_forward<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, hidden: &Tensor<S>, hx: &Tensor<S>,
    bias_i: &Tensor<S>, bias_h: &Tensor<S>, hy: &mut Tensor<S>,
    workspace: &mut Tensor<S>, #[comptime] has_bias: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < hx.len() {
        let width = hx.shape(1);
        let column = p % width;
        let base = (p / width) * 3usize * width + column;
        let r = sigmoid::<F>(gate::<S, F>(input, hidden, bias_i, bias_h, base, column, has_bias));
        let i = sigmoid::<F>(gate::<S, F>(input, hidden, bias_i, bias_h, base + width, column + width, has_bias));
        let mut bin = F::new(0.0);
        let mut bhn = F::new(0.0);
        if comptime!(has_bias) {
            bin = F::cast_from(bias_i[address(bias_i, column + 2usize * width)]);
            bhn = F::cast_from(bias_h[address(bias_h, column + 2usize * width)]);
        }
        let hn = F::cast_from(hidden[address(hidden, base + 2usize * width)]) + bhn;
        let n = (F::cast_from(input[address(input, base + 2usize * width)]) + bin + r * hn).tanh();
        let previous = hx[address(hx, p)];
        hy[address(hy, p)] = S::cast_from(n + i * (F::cast_from(previous) - n));
        let saved = (p / width) * 5usize * width + column;
        workspace[address(workspace, saved)] = S::cast_from(r);
        workspace[address(workspace, saved + width)] = S::cast_from(i);
        workspace[address(workspace, saved + 2usize * width)] = S::cast_from(n);
        workspace[address(workspace, saved + 3usize * width)] = previous;
        workspace[address(workspace, saved + 4usize * width)] = S::cast_from(hn);
    }
}

#[ruda(launch)]
fn gru_backward<S: Float + RudaElement, F: Float + RudaElement>(
    grad_hy: &Tensor<S>, workspace: &Tensor<S>, grad_input: &mut Tensor<S>,
    grad_hidden: &mut Tensor<S>, grad_hx: &mut Tensor<S>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < grad_hy.len() {
        let width = grad_hy.shape(1);
        let column = p % width;
        let saved = (p / width) * 5usize * width + column;
        let r = F::cast_from(workspace[address(workspace, saved)]);
        let i = F::cast_from(workspace[address(workspace, saved + width)]);
        let n = F::cast_from(workspace[address(workspace, saved + 2usize * width)]);
        let hx = F::cast_from(workspace[address(workspace, saved + 3usize * width)]);
        let hn = F::cast_from(workspace[address(workspace, saved + 4usize * width)]);
        let g = F::cast_from(grad_hy[address(grad_hy, p)]);
        let one = F::new(1.0);
        let gi = g * (hx - n) * (one - i) * i;
        let ghx = g * i;
        let gin = g * (one - i) * (one - n * n);
        let ghn = gin * r;
        let gr = gin * hn * (one - r) * r;
        let base = (p / width) * 3usize * width + column;
        grad_input[address(grad_input, base)] = S::cast_from(gr);
        grad_input[address(grad_input, base + width)] = S::cast_from(gi);
        grad_input[address(grad_input, base + 2usize * width)] = S::cast_from(gin);
        grad_hidden[address(grad_hidden, base)] = S::cast_from(gr);
        grad_hidden[address(grad_hidden, base + width)] = S::cast_from(gi);
        grad_hidden[address(grad_hidden, base + 2usize * width)] = S::cast_from(ghn);
        grad_hx[address(grad_hx, p)] = S::cast_from(ghx);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 443..=446));
    assert_eq!(xs.len(), match op { 443 => 8, 444 | 445 => 7, _ => 5 });
    assert_eq!(ints.len(), 1);
    let max_flags = match op { 444 => 3, 446 => 0, _ => 1 };
    assert!((0..=max_flags).contains(&ints[0]));
    let state = &xs[if op == 446 { 0 } else { 2 }];
    assert_eq!(state.shape.len(), 2);
    assert!(matches!(state.dtype, 0 | 1 | 2 | 9));
    for value in xs { assert_eq!(value.dtype, state.dtype); }
    let factor = if op <= 444 { 4usize } else { 3usize };
    let has_bias = ints[0] != 0;
    if op == 443 || op == 445 {
        assert_eq!(xs[0].shape.len(), 2);
        assert_eq!(xs[0].shape, xs[1].shape);
        assert_eq!(xs[0].len / factor, state.len);
        assert_eq!(xs[5].shape, state.shape);
        if op == 443 {
            assert_eq!(xs[6].shape, state.shape);
            assert_eq!(xs[7].shape, xs[0].shape);
        } else { assert_eq!(xs[6].shape, vec![state.shape[0], state.shape[1] * 5]); }
        if has_bias {
            assert_eq!(xs[3].shape, vec![xs[0].shape[1]]);
            assert_eq!(xs[4].shape, xs[3].shape);
            assert!(state.len == 0 || xs[3].len >= state.shape[1] * factor);
        }
    } else if op == 444 {
        assert_ne!(ints[0], 0);
        for index in [0usize, 1, 3, 6] { assert_eq!(xs[index].shape, state.shape); }
        assert_eq!(xs[4].shape.len(), 2);
        assert_eq!(xs[4].len, state.len * 4);
        assert_eq!(xs[5].shape, xs[4].shape);
    } else {
        assert_eq!(xs[1].shape, vec![state.shape[0], state.shape[1] * 5]);
        assert_eq!(xs[2].shape, vec![state.shape[0], state.shape[1] * 3]);
        assert_eq!(xs[3].shape, xs[2].shape);
        assert_eq!(xs[4].shape, state.shape);
    }
    if state.len == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(state.len.div_ceil(512)).expect("RNN cell launch overflow"), 1, 1);
    let block = RudaDim::new_1d(512);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        match op {
            443 => lstm_forward::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(), xs[6].arg(), xs[7].arg(), has_bias),
            444 => lstm_backward::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(), xs[6].arg(),
                ints[0] & 1 != 0, ints[0] & 2 != 0),
            445 => gru_forward::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(), xs[6].arg(), has_bias),
            446 => gru_backward::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg()),
            _ => unreachable!(),
        }
    }}; }
    match state.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32),
        9 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
