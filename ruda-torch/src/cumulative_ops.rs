// ATen scan combine/order rules; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn combine<S: Numeric, A: Numeric>(
    ar: S, ai: S, br: S, bi: S,
    #[comptime] product: bool, #[comptime] complex: bool,
) -> (S, S) {
    let x = A::cast_from(ar);
    let y = A::cast_from(br);
    let mut real = x + y;
    let mut imag = A::cast_from(0u32);
    if comptime!(product) {
        real = x * y;
        if comptime!(complex) {
            let u = A::cast_from(ai);
            let v = A::cast_from(bi);
            real = x * y - u * v;
            imag = x * v + u * y;
        }
    } else if comptime!(complex) {
        imag = A::cast_from(ai) + A::cast_from(bi);
    }
    (S::cast_from(real), S::cast_from(imag))
}

#[ruda(launch)]
fn outer<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    ar: &Tensor<S>, ai: &Tensor<S>, re: &mut Tensor<S>, im: &mut Tensor<S>,
    #[comptime] product: bool, #[comptime] complex: bool,
) {
    let row = ABSOLUTE_POS as usize;
    let width = ar.shape(1);
    let inner = ar.shape(2);
    if row < ar.shape(0) * inner {
        let base = row / inner * width * inner + row % inner;
        let mut real = S::cast_from(0u32);
        let mut imag = S::cast_from(0u32);
        if comptime!(product) { real = S::cast_from(1u32); }
        for position in 0..width {
            let index = base + position * inner;
            let mut value_i = S::cast_from(0u32);
            if comptime!(complex) { value_i = ai[address(ai, index)]; }
            let (r, i) = combine::<S, A>(real, imag, ar[address(ar, index)], value_i, product, complex);
            real = r;
            imag = i;
            re[address(re, index)] = real;
            if comptime!(complex) { im[address(im, index)] = imag; }
        }
    }
}

#[ruda(launch)]
fn outer_vector<S:Numeric+RudaElement,A:Numeric+RudaElement>(input:&Array<Vector<S,Const<4>>>,out:&mut Array<Vector<S,Const<4>>>,
    width:usize,inner:usize,#[comptime] product:bool)
{
    let row=ABSOLUTE_POS as usize;
    if row<out.len()/width {
        let base=row/(inner/4)*width*(inner/4)+row%(inner/4);
        let mut value=Vector::<S,Const<4>>::new(S::cast_from(0u32));
        if comptime!(product) {value=Vector::new(S::cast_from(1u32));}
        for position in 0usize..width {
            let index=base+position*(inner/4);
            let x=Vector::<A,Const<4>>::cast_from(value);
            let y=Vector::<A,Const<4>>::cast_from(input[index]);
            if comptime!(product) {value=Vector::cast_from(x*y);}else {value=Vector::cast_from(x+y);}
            out[index]=value;
        }
    }
}

#[ruda(launch)]
fn inner<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    ar: &Tensor<S>, ai: &Tensor<S>, re: &mut Tensor<S>, im: &mut Tensor<S>,
    #[comptime] product: bool, #[comptime] complex: bool,
    #[comptime] threads: usize, #[comptime] single: bool,
) {
    let row = (RUDA_POS_X + RUDA_POS_Y * RUDA_COUNT_X) as usize;
    let lane = UNIT_POS as usize;
    if row < ar.shape(0) {
        let width = i64::cast_from(ar.shape(1));
        let base = row * ar.shape(1);
        let mut identity = S::cast_from(0u32);
        if comptime!(product) { identity = S::cast_from(1u32); }
        let zero = S::cast_from(0u32);
        let mut real = SharedMemory::<S>::new(threads * 2);
        let mut imag = SharedMemory::<S>::new(threads * 2);
        let mut carry_r = identity;
        let mut carry_i = zero;
        let mut tile = 0i64;
        while tile < width {
            let first = tile + i64::cast_from(lane);
            let second = first + i64::cast_from(threads);
            real[lane] = identity;
            real[lane + threads] = identity;
            imag[lane] = zero;
            imag[lane + threads] = zero;
            if first < width {
                let index = base + usize::cast_from(first);
                real[lane] = ar[address(ar, index)];
                if comptime!(complex) { imag[lane] = ai[address(ai, index)]; }
            }
            if second < width {
                let index = base + usize::cast_from(second);
                real[lane + threads] = ar[address(ar, index)];
                if comptime!(complex) { imag[lane + threads] = ai[address(ai, index)]; }
            }
            if lane == 0 {
                let mut include = true;
                if comptime!(single) { include = tile > 0i64; }
                if include {
                    let (r, i) = combine::<S, A>(real[0], imag[0], carry_r, carry_i, product, complex);
                    real[0] = r;
                    imag[0] = i;
                }
            }
            sync_ruda();
            let mut step = 1usize;
            while step <= threads {
                let start = lane / step * (step * 2) + step;
                let to = start + lane % step;
                let from = start - 1;
                let (r, i) = combine::<S, A>(real[to], imag[to], real[from], imag[from], product, complex);
                real[to] = r;
                imag[to] = i;
                sync_ruda();
                step *= 2;
            }
            if first < width {
                let index = base + usize::cast_from(first);
                re[address(re, index)] = real[lane];
                if comptime!(complex) { im[address(im, index)] = imag[lane]; }
            }
            if second < width {
                let index = base + usize::cast_from(second);
                re[address(re, index)] = real[lane + threads];
                if comptime!(complex) { im[address(im, index)] = imag[lane + threads]; }
            }
            carry_r = real[threads * 2 - 1];
            carry_i = imag[threads * 2 - 1];
            sync_ruda();
            tile += i64::cast_from(threads * 2);
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 405 | 406));
    assert_eq!(xs.len(), 4);
    assert_eq!(ints.len(), 2);
    assert!(ints.iter().all(|&n| matches!(n, 0 | 1)));
    let input = &xs[0];
    assert_eq!(input.shape.len(), 3);
    assert!(matches!(input.dtype, 0 | 1 | 2 | 4..=9));
    for value in &xs[1..] {
        assert_eq!(value.shape, input.shape);
        assert_eq!(value.dtype, input.dtype);
    }
    let complex = ints[0] != 0;
    let parallel = ints[1] != 0;
    if complex { assert!(matches!(input.dtype, 0 | 9)); }
    if input.len == 0 { return; }
    let width = input.shape[1];
    let rows = input.shape[0] * input.shape[2];
    let single = input.len == width;
    let c = client();
    if op == 405 && parallel && !complex && matches!(input.dtype,4..=8)
        && input.contiguous() && xs[2].contiguous() {
        let (nodes,_workspace) = sequence_scan::prepare(&c,input,&xs[2],14,1);
        let launches = nodes.len();
        for node in nodes {
            let (task,count,args,origin) = node.into_parts();origin.launch(task,count,args);
        }
        LAUNCHES.fetch_add(launches as u64, Ordering::Relaxed);finish_dispatch(&c);return;
    }
    let mut threads = 256usize;
    let vector=!parallel && !complex && input.shape[2]%4==0
        && [input,&xs[2]].iter().all(|v|v.contiguous()
            && v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0);
    let grid;
    if parallel {
        assert_eq!(input.shape[2], 1);
        let log_x = usize::BITS - (width - 1).leading_zeros();
        let log_y = usize::BITS - (rows - 1).leading_zeros();
        let log_threads = (9u32.wrapping_add(log_x.wrapping_sub(log_y)) / 2).clamp(4, 9);
        threads = 1usize << log_threads;
        let count = u32::try_from(rows).expect("cumulative row overflow");
        let x = count.min(1 << 20);
        grid = RudaCount::Static(x, count.div_ceil(x), 1);
    } else {
        grid = RudaCount::Static(u32::try_from((rows/if vector {4}else {1}).div_ceil(threads)).expect("cumulative launch overflow"), 1, 1);
    }
    let block = RudaDim::new_1d(threads as u32);
    macro_rules! run { ($s:ty, $a:ty) => { unsafe {
        if parallel {
            inner::launch::<$s, $a, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), op == 406, complex, threads, single);
        } else if vector {
            outer_vector::launch::<$s,$a,CudaRuntime>(&c,grid,block,xs[0].array_arg(),xs[2].array_arg(),width,input.shape[2],op==406);
        } else {
            outer::launch::<$s, $a, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), op == 406, complex);
        }
    }}; }
    match input.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32),
        4 => run!(i64, u64), 5 => run!(i32, u64), 6 => run!(i16, u64),
        7 => run!(i8, u64), 8 => run!(u8, u64), 9 => run!(f64, f64),
        _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
