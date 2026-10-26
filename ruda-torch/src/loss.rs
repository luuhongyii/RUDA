// ATen CUDA loss formulas; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn stored<S:Float,F:Float>(x:F)->F {F::cast_from(S::cast_from(x))}

#[ruda(launch)]
fn pointwise<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<S>,grad:&Tensor<S>,out:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,boundary:F,norm:F,#[comptime] op:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=F::cast_from(a[address(a,p)]);let t=F::cast_from(target[address(target,p)]);
        let g=F::cast_from(grad[address(grad,p)]);let one=F::new(1.0);let zero=F::new(0.0);
        let d=stored::<S,F>(x-t);let b=stored::<S,F>(boundary);let n=stored::<S,F>(norm);
        let mut y=zero;
        if comptime!(op==293) {y=d*d;}
        else if comptime!(op==294) {y=stored::<S,F>(n*d)*g;}
        else if comptime!(op==295 || op==297) {
            let z=d.abs();let half_b=stored::<S,F>(F::new(0.5)*b);
            if z<b {y=stored::<S,F>(stored::<S,F>(F::new(0.5)*z)*z);if comptime!(op==295) {y=y/b;}}
            else {y=stored::<S,F>(z-half_b);if comptime!(op==297) {y=b*y;}}
        } else if comptime!(op==296 || op==298) {
            if d< -b {y=stored::<S,F>(-n*g);if comptime!(op==298) {y=y*b;}}
            else if d>b {y=stored::<S,F>(n*g);if comptime!(op==298) {y=y*b;}}
            else {y=stored::<S,F>(stored::<S,F>(n*d)*g);if comptime!(op==296) {y=y/b;}}
        } else if comptime!(op==299) {
            if !(x>=zero && x<=one && t>=zero && t<=one) {invalid[0].store(1u32);}
            let mut lx=stored::<S,F>(x.ln());let mut ln=stored::<S,F>((-x).log1p());
            if lx<F::new(-100.0) {lx=F::new(-100.0);}if ln<F::new(-100.0) {ln=F::new(-100.0);}
            y=stored::<S,F>(stored::<S,F>(t-one)*ln)-stored::<S,F>(t*lx);
        } else if comptime!(op==300) {
            let mut den=stored::<S,F>(stored::<S,F>(one-x)*x);let eps=stored::<S,F>(F::cast_from(1e-12f32));
            if den<eps {den=eps;}y=stored::<S,F>(g*d)/den;
        } else if comptime!(op==301) {
            let z=stored::<S,F>(-x*t);y=stored::<S,F>(z.exp()).log1p();
        } else if comptime!(op==302) {
            let z=stored::<S,F>(stored::<S,F>(-t*x).exp());
            y=stored::<S,F>(stored::<S,F>(t*z)*(-norm));
            y=stored::<S,F>(y/stored::<S,F>(one+z))*g;
        }
        out[address(out,p)]=S::cast_from(y);
    }
}

#[ruda(launch)]
fn nll_unreduced<S:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,out:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,#[comptime] weighted:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<target.len() {
        let t=target[address(target,p)];let mut y=S::new(0.0);
        if t!=ignore {
            if t<0i64 || t>=i64::cast_from(a.shape(1)) {invalid[0].store(1u32);}
            else {
                let spatial=a.shape(2);let offset=p/spatial*a.stride(0)+usize::cast_from(t)*a.stride(1)+(p%spatial)*a.stride(2);
                let mut w=S::new(1.0);if comptime!(weighted) {w=weight[address(weight,usize::cast_from(t))];}
                y= -w*a[offset];
            }
        }
        out[address(out,p)]=y;
    }
}

#[ruda(launch)]
fn nll_single<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,out:&mut Tensor<S>,total:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,#[comptime] weighted:bool,#[comptime] mean:bool) {
    let t=target[0];let mut y=S::new(0.0);let mut w=S::new(0.0);
    if t!=ignore {
        if t<0i64 || t>=i64::cast_from(a.shape(1)) {invalid[0].store(1u32);}
        else {
            w=S::new(1.0);if comptime!(weighted) {w=weight[address(weight,usize::cast_from(t))];}
            let x=a[usize::cast_from(t)*a.stride(1)];
            y= -w*x;
            if comptime!(mean) {y= -x;if w==S::new(0.0) {y=S::cast_from(F::new(f32::NAN));}}
        }
    }
    out[0]=y;total[0]=w;
}

#[ruda(launch)]
fn nll_reduce<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,partial:&mut Tensor<F>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,blocks:usize,#[comptime] threads:usize,#[comptime] spatial:bool,#[comptime] weighted:bool) {
    let lane=UNIT_POS as usize;let block=RUDA_POS_X as usize;
    let mut shared_loss=SharedMemory::<F>::new(threads);let mut shared_weight=SharedMemory::<F>::new(threads);
    let mut start=lane;let mut end=target.len();let mut step=usize::cast_from(threads);
    if comptime!(spatial) {let sample=block/blocks;start=sample*a.shape(2)+(block%blocks)*threads+lane;end=(sample+1)*a.shape(2);step=threads*blocks;}
    let mut loss=F::new(0.0);let mut total=F::new(0.0);let mut p=start;
    while p<end {
        let t=target[address(target,p)];
        if t!=ignore {
            if t<0i64 || t>=i64::cast_from(a.shape(1)) {invalid[0].store(1u32);}
            else {
                let width=a.shape(2);let offset=p/width*a.stride(0)+usize::cast_from(t)*a.stride(1)+(p%width)*a.stride(2);
                let mut w=S::new(1.0);if comptime!(weighted) {w=weight[address(weight,usize::cast_from(t))];}
                loss-=F::cast_from(a[offset]*w);total+=F::cast_from(w);
            }
        }
        p+=step;
    }
    shared_loss[lane]=loss;shared_weight[lane]=total;sync_ruda();
    let mut stride=usize::cast_from(threads/2);
    while stride>0 {if lane<stride {shared_loss[lane]=shared_loss[lane]+shared_loss[lane+stride];shared_weight[lane]=shared_weight[lane]+shared_weight[lane+stride];}sync_ruda();stride/=2;}
    if lane==0 {
        let mut value=shared_loss[0];let mut count=shared_weight[0];
        if comptime!(spatial) {value=stored::<S,F>(value);count=stored::<S,F>(count);}
        partial[block*2]=value;partial[block*2+1]=count;
    }
}

#[ruda(launch)]
fn nll_finish<S:Float+RudaElement,F:Float+RudaElement>(partial:&Tensor<F>,out:&mut Tensor<S>,total:&mut Tensor<S>,#[comptime] spatial:bool,#[comptime] mean:bool) {
    let mut value=F::new(0.0);let mut count=F::new(0.0);
    for i in 0..partial.len()/2 {
        value+=partial[i*2];count+=partial[i*2+1];
        if comptime!(spatial) {value=stored::<S,F>(value);count=stored::<S,F>(count);}
    }
    total[0]=S::cast_from(count);if comptime!(mean) {value=value/count;}out[0]=S::cast_from(value);
}

#[ruda(launch)]
fn nll_backward<S:Float+RudaElement>(target:&Tensor<i64>,weight:&Tensor<S>,grad:&Tensor<S>,total:&Tensor<S>,out:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,#[comptime] weighted:bool,#[comptime] reduction:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<target.len() {
        let t=target[address(target,p)];
        if t!=ignore {
            if t<0i64 || t>=i64::cast_from(out.shape(1)) {invalid[0].store(1u32);}
            else {
                let mut g=grad[0];if comptime!(reduction==0) {g=grad[address(grad,p)];}
                if comptime!(reduction==1) {g=g/total[0];}
                g= -g;if comptime!(weighted) {g=weight[address(weight,usize::cast_from(t))]*g;}
                let width=out.shape(2);let offset=p/width*out.stride(0)+usize::cast_from(t)*out.stride(1)+(p%width)*out.stride(2);
                out[offset]=g;
            }
        }
    }
}

fn grid(len:usize)->RudaCount {RudaCount::Static(u32::try_from(len.div_ceil(128)).expect("loss grid overflow"),1,1)}
fn status(c:&ComputeClient<CudaRuntime>)->View {
    let value=View::packed(c.empty(4),vec![1],5);let bytes=value.byte_view();
    unsafe {kernels::fill_bytes::launch::<CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(4),bytes.arg(),0,0);}
    value
}
fn check(c:&ComputeClient<CudaRuntime>,value:&View,message:&str) {
    let bytes=c.read_one(value.handle.clone()).expect("loss validation readback failed");DOWNLOAD.fetch_add(4,Ordering::Relaxed);
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()),0,"{message}");
}

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn cuda_exp2(value: f32) -> f32 { f32::powf(2.0f32, value) }

#[ruda]
pub(crate) fn cuda_exp(value: f32, add: f32) -> f32 {
    let mut fraction = fma(value, f32::reinterpret(0x3bbb989du32), 0.5f32);
    if !(fraction > 0.0f32) { fraction = 0.0f32; }
    if fraction > 1.0f32 { fraction = 1.0f32; }
    let integer = u32::cast_from(f64::cast_from(fraction) * 252.0f64);
    let encoded = 12582913.0f32 + f32::cast_from(integer);
    let exponent = encoded - 12583039.0f32;
    let reduced = fma(value, f32::reinterpret(0x3fb8aa3bu32), -exponent);
    let reduced = fma(value, f32::reinterpret(0x32a57060u32), reduced);
    let scale = f32::reinterpret(u32::reinterpret(encoded) << 23u32);
    fma(cuda_exp2(reduced), scale, add)
}

#[ruda]
fn cuda_log(value: f32) -> f32 {
    let mut normalized = value;
    let mut offset = 0.0f32;
    if value < f32::reinterpret(0x00800000u32) {
        normalized *= 8388608.0f32;
        offset = -23.0f32;
    }
    let bits = u32::reinterpret(normalized);
    let adjustment = (bits - 1059760811u32) & 0xff800000u32;
    let mantissa = f32::reinterpret(bits - adjustment);
    let exponent = fma(f32::cast_from(i32::reinterpret(adjustment)), f32::reinterpret(0x34000000u32), offset);
    let t = mantissa - 1.0f32;
    let mut p = fma(f32::reinterpret(0xbe055027u32), t, f32::reinterpret(0x3e1039f6u32));
    p = fma(p, t, f32::reinterpret(0xbdf8cdccu32));
    p = fma(p, t, f32::reinterpret(0x3e0f2955u32));
    p = fma(p, t, f32::reinterpret(0xbe2ad8b9u32));
    p = fma(p, t, f32::reinterpret(0x3e4ced0bu32));
    p = fma(p, t, f32::reinterpret(0xbe7fff22u32));
    p = fma(p, t, f32::reinterpret(0x3eaaaa78u32));
    p = fma(p, t, -0.5f32);
    let polynomial = fma(t * p, t, t);
    let mut result = fma(exponent, f32::reinterpret(0x3f317218u32), polynomial);
    if bits >= 0x7f800000u32 {
        result = fma(normalized, f32::new(f32::INFINITY), f32::new(f32::INFINITY));
    }
    if normalized == 0.0f32 { result = f32::new(f32::NEG_INFINITY); }
    result
}

#[ruda]
fn cross_entropy_block_reduce(value: f32, #[comptime] maximum: bool) -> f32 {
    let thread = UNIT_POS as usize;
    let mut total = value;
    #[unroll]
    for stage in 0usize..5 {
        let other = plane_shuffle_down(total, (16usize >> stage) as u32);
        if comptime!(maximum) { total = f32::max(total, other); }
        else { total += other; }
    }
    let mut partial = SharedMemory::<f32>::new(32usize);
    if thread % 32 == 0 { partial[thread / 32] = total; }
    sync_ruda();
    if thread < 32 {
        total = partial[thread];
        #[unroll]
        for stage in 0usize..5 {
            let other = plane_shuffle_down(total, (16usize >> stage) as u32);
            if comptime!(maximum) { total = f32::max(total, other); }
            else { total += other; }
        }
    }
    if thread == 0 { partial[0] = total; }
    sync_ruda();
    partial[0]
}

#[ruda(launch)]
fn cross_entropy_cuda(
    x: &Tensor<bf16>, targets: &Tensor<i64>, losses: &mut Tensor<f32>,
    stats: &mut Tensor<f32>, invalid: &mut Tensor<Atomic<u32>>, ignore: i64,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let target = targets[row];
    let width = x.shape(1);
    let vector_end = width / 4096 * 4096;
    let mut maximum = f32::new(f32::MIN);
    let mut log_sum = 0.0f32;
    if target != ignore {
        let mut col = thread * 4;
        while col < vector_end {
            #[unroll]
            for j in 0usize..4 { maximum = f32::max(maximum, f32::cast_from(x[row * width + col + j])); }
            col += 4096;
        }
        col = vector_end + thread;
        while col < width {
            maximum = f32::max(maximum, f32::cast_from(x[row * width + col]));
            col += 1024;
        }
        maximum = cross_entropy_block_reduce(maximum, true);
        let mut sum = 0.0f32;
        col = thread * 4;
        while col < vector_end {
            #[unroll]
            for j in 0usize..4 { sum = cuda_exp(f32::cast_from(x[row * width + col + j]) - maximum, sum); }
            col += 4096;
        }
        col = vector_end + thread;
        while col < width {
            sum = cuda_exp(f32::cast_from(x[row * width + col]) - maximum, sum);
            col += 1024;
        }
        let total = cross_entropy_block_reduce(sum, false);
        if thread == 0 { log_sum = cuda_log(total); }
    }
    if thread == 0 {
        let mut loss = 0.0f32;
        if target != ignore {
            if target < 0i64 || target >= i64::cast_from(width) { invalid[0].store(1u32); }
            else { loss = (maximum - f32::cast_from(x[row * width + usize::cast_from(target)])) + log_sum; }
        } else { maximum = 0.0f32; }
        stats[row * 2] = maximum;
        stats[row * 2 + 1] = log_sum;
        losses[row] = loss;
    }
}

#[ruda(launch)]
fn cross_entropy_tiles<S: Float + RudaElement>(
    x: &Tensor<S>, targets: &Tensor<i64>, partial: &mut Tensor<f32>, ignore: i64,
) {
    let warp = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let tiles = partial.shape(1);
    let row = warp / tiles;
    let tile = warp % tiles;
    let mut active = false;
    if row < x.shape(0) { active = targets[row] != ignore; }
    if active {
        let width = x.shape(1);
        let mut maximum = f32::new(f32::NEG_INFINITY);
        let mut cached = Array::<f32>::new(32usize);
        #[unroll]
        for j in 0usize..32usize {
            let col = tile * 1024 + lane + j * 32;
            let mut value = f32::new(f32::NEG_INFINITY);
            if col < width {
                value = f32::cast_from(x[row * width + col]);
                maximum = f32::max(maximum, value);
            }
            cached[j] = value;
        }
        maximum = plane_max(maximum);
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0usize..32usize {
            let col = tile * 1024 + lane + j * 32;
            if col < width && maximum != f32::new(f32::NEG_INFINITY) {
                sum += (cached[j] - maximum).exp();
            }
        }
        sum = plane_sum(sum);
        if lane == 0 {
            partial[warp * 2] = maximum;
            partial[warp * 2 + 1] = sum;
        }
    } else if row < x.shape(0) && lane == 0 {
        partial[warp * 2] = 0.0f32;
        partial[warp * 2 + 1] = 0.0f32;
    }
}

#[ruda(launch)]
fn cross_entropy_finish<S: Float + RudaElement>(
    x: &Tensor<S>, targets: &Tensor<i64>, partial: &Tensor<f32>,
    losses: &mut Tensor<f32>, stats: &mut Tensor<f32>, invalid: &mut Tensor<Atomic<u32>>,
    ignore: i64,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let mut target = ignore;
    if row < targets.len() { target = targets[row]; }
    if target != ignore {
        let tiles = partial.shape(1);
        let mut tile = lane;
        let mut maximum = f32::new(f32::NEG_INFINITY);
        while tile < tiles {
            maximum = f32::max(maximum, partial[(row * tiles + tile) * 2]);
            tile += 32;
        }
        maximum = plane_max(maximum);
        let mut sum = 0.0f32;
        tile = lane;
        while tile < tiles {
            let pos = (row * tiles + tile) * 2;
            sum += partial[pos + 1] * (partial[pos] - maximum).exp();
            tile += 32;
        }
        let log_sum = plane_sum(sum).ln();
        if lane == 0 {
            stats[row * 2] = maximum;
            stats[row * 2 + 1] = log_sum;
            let mut loss = 0.0f32;
            if target < 0i64 || target >= i64::cast_from(x.shape(1)) {
                invalid[0].store(1u32);
            } else {
                loss = (maximum - f32::cast_from(x[row * x.shape(1) + usize::cast_from(target)])) + log_sum;
            }
            losses[row] = loss;
        }
    } else if row < targets.len() && lane == 0 {
        stats[row * 2] = 0.0f32;
        stats[row * 2 + 1] = 0.0f32;
        losses[row] = 0.0f32;
    }
}

#[ruda(launch)]
fn cross_entropy_backward<S: Float + RudaElement>(
    x: &Tensor<S>, targets: &Tensor<i64>, stats: &Tensor<f32>,
    grad: &Tensor<f32>, dx: &mut Tensor<S>, ignore: i64, #[comptime] cuda: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < x.len() {
        let width = x.shape(1);
        let row = pos / width;
        let target = targets[row];
        let mut value = 0.0f32;
        if target != ignore {
            let log_probability = (f32::cast_from(x[pos]) - stats[row * 2]) - stats[row * 2 + 1];
            if comptime!(cuda) { value = cuda_exp(log_probability, 0.0f32); }
            else { value = log_probability.exp(); }
            if i64::cast_from(pos % width) == target { value -= 1.0f32; }
            value *= grad[address(grad, row)];
        }
        dx[pos] = S::cast_from(value);
    }
}

pub(super) fn dispatch_fused_cross_entropy(op: u32, xs: &[View], ints: &[i64]) {
    assert_eq!(ints.len(), 1);
    assert_eq!(xs.len(), if op == 650 { 4 } else { 5 });
    for (index, x) in xs.iter().enumerate() {
        if op == 651 && index == 3 { continue; }
        let mut stride = 1usize;
        for (&dim, &actual) in x.shape.iter().zip(&x.strides).rev() {
            assert!(dim <= 1 || actual == stride, "cross entropy requires contiguous storage");
            stride = stride.checked_mul(dim).expect("loss shape overflow");
        }
    }
    let x = &xs[0]; let target = &xs[1];
    assert_eq!(x.shape.len(), 2);
    let rows = x.shape[0]; let width = x.shape[1];
    assert!(x.dtype <= 2 && width > 0);
    assert_eq!(target.dtype, 4); assert_eq!(target.shape, vec![rows]);
    let (losses, stats) = if op == 650 { (&xs[2], &xs[3]) } else { (&xs[3], &xs[2]) };
    assert_eq!((losses.dtype, &losses.shape), (0, &vec![rows]));
    assert_eq!((stats.dtype, &stats.shape), (0, &vec![rows, 2]));
    if op == 651 { assert_eq!(xs[4].shape, x.shape); assert_eq!(xs[4].dtype, x.dtype); }
    if rows == 0 { return; }
    let c = client(); let dim = RudaDim::new_1d(128);
    let cuda = x.dtype == 2 && width >= 32768 && width % 4 == 0 && super::matmul::cuda_bf16_precision();
    if op == 650 && cuda {
        let invalid = status(&c);
        unsafe {
            cross_entropy_cuda::launch::<CudaRuntime>(&c,
                RudaCount::Static(u32::try_from(rows).expect("loss grid overflow"),1,1),RudaDim::new_1d(1024),
                x.arg(),target.arg(),losses.arg(),stats.arg(),invalid.arg(),ints[0]);
        }
        check(&c, &invalid, "cross entropy target is out of bounds");
    } else if op == 650 {
        let tiles = width.div_ceil(1024);
        let elements = rows.checked_mul(tiles).and_then(|n| n.checked_mul(2)).expect("loss partial overflow");
        let partial = View::packed(c.empty(elements.checked_mul(4).expect("loss allocation overflow")), vec![rows, tiles, 2], 0);
        let invalid = status(&c);
        let work = rows.checked_mul(tiles).and_then(|n| n.checked_mul(32)).expect("loss grid overflow");
        macro_rules! run { ($s:ty) => { unsafe {
            cross_entropy_tiles::launch::<$s,CudaRuntime>(&c,grid(work),dim,
                x.arg(),target.arg(),partial.arg(),ints[0]);
            cross_entropy_finish::launch::<$s,CudaRuntime>(&c,grid(rows.checked_mul(32).expect("loss grid overflow")),dim,
                x.arg(),target.arg(),partial.arg(),losses.arg(),stats.arg(),invalid.arg(),ints[0]);
        } }; }
        match x.dtype { 0=>run!(f32), 1=>run!(f16), 2=>run!(bf16), _=>unreachable!() }
        check(&c, &invalid, "cross entropy target is out of bounds");
    } else {
        macro_rules! run { ($s:ty) => { unsafe {
            cross_entropy_backward::launch::<$s,CudaRuntime>(&c,grid(x.len),dim,
                x.arg(),target.arg(),stats.arg(),losses.arg(),xs[4].arg(),ints[0],cuda);
        } }; }
        match x.dtype { 0=>run!(f32), 1=>run!(f16), 2=>run!(bf16), _=>unreachable!() }
    }
}

fn nll_dispatch<S:Float+RudaElement,F:Float+RudaElement>(c:&ComputeClient<CudaRuntime>,op:u32,xs:&[View],ints:&[i64]) {
    let a=&xs[0];let target=&xs[1];let weight=&xs[2];let out=xs.last().unwrap();
    let reduction=ints[0] as u32;let ignore=ints[1];let mode=ints[2];let weighted=ints[3]!=0;
    let invalid=status(c);let dim=RudaDim::new_1d(128);
    if op==304 {
        if target.len>0 {unsafe {nll_backward::launch::<S,CudaRuntime>(c,grid(target.len),dim,target.arg(),weight.arg(),xs[3].arg(),xs[4].arg(),out.arg(),invalid.arg(),ignore,weighted,reduction);}}
    } else if mode==0 {
        unsafe {nll_single::launch::<S,F,CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),a.arg(),target.arg(),weight.arg(),xs[3].arg(),xs[4].arg(),invalid.arg(),ignore,weighted,reduction==1);}
    } else if reduction==0 {
        if target.len>0 {unsafe {nll_unreduced::launch::<S,CudaRuntime>(c,grid(target.len),dim,a.arg(),target.arg(),weight.arg(),xs[3].arg(),invalid.arg(),ignore,weighted);}}
    } else {
        let spatial=mode==2;let blocks=if spatial {a.shape[2].div_ceil(1024).div_euclid(128).max(1)}else{1};
        let count=if spatial {a.shape[0]*blocks}else{1};
        let threads=if spatial {1024}else{((target.len.max(16)/16) as f64).log2().round().exp2().clamp(32.0,1024.0) as usize};
        let partial=View::packed(c.empty((count*2*std::mem::size_of::<F>()).max(1)),vec![count*2],if a.dtype==9 {9}else{0});
        if count>0 {unsafe {nll_reduce::launch::<S,F,CudaRuntime>(c,RudaCount::Static(u32::try_from(count).expect("NLL grid overflow"),1,1),RudaDim::new_1d(threads as u32),a.arg(),target.arg(),weight.arg(),partial.arg(),invalid.arg(),ignore,blocks,threads,spatial,weighted);}}
        unsafe {nll_finish::launch::<S,F,CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),partial.arg(),xs[3].arg(),xs[4].arg(),spatial,reduction==1);}
    }
    check(c,&invalid,"NLL target is out of bounds");
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    let c=client().fixed_execution_queue();let a=&xs[0];
    if op<=302 {
        assert_eq!(xs.len(),4);assert_eq!(values.len(),2);
        for x in xs {assert_eq!(a.shape,x.shape);assert_eq!(a.dtype,x.dtype);}
        if a.len==0 {return;}
        let invalid=if op==299 {status(&c)}else{View::packed(c.empty(4),vec![1],5)};
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {pointwise::launch::<$s,$f,CudaRuntime>(&c,grid(a.len),RudaDim::new_1d(128),a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),invalid.arg(),values[0] as $f,values[1] as $f,op);}}}
        match a.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>panic!("loss requires real floating input")}
        if op==299 {check(&c,&invalid,"all BCE input and target elements must be between 0 and 1");}
    } else {
        assert_eq!(xs.len(),if op==303 {5}else{6});assert_eq!(ints.len(),4);
        assert!((0..=2).contains(&ints[0]) && (0..=2).contains(&ints[2]));
        assert_eq!(a.shape.len(),3);assert_eq!(xs[1].dtype,4);assert_eq!(xs[1].len,a.shape[0]*a.shape[2]);
        for x in &xs[2..] {assert_eq!(a.dtype,x.dtype);}
        if ints[3]!=0 {assert_eq!(xs[2].len,a.shape[1]);}
        if op==304 {assert_eq!(xs[5].shape,a.shape);assert_eq!(xs[4].len,1);assert_eq!(xs[3].len,if ints[0]==0 {xs[1].len}else{1});}
        else {assert_eq!(xs[4].len,1);assert_eq!(xs[3].len,if ints[0]==0 && ints[2]!=0 {xs[1].len}else{1});}
        match a.dtype {0=>nll_dispatch::<f32,f32>(&c,op,xs,ints),1=>nll_dispatch::<f16,f32>(&c,op,xs,ints),2=>nll_dispatch::<bf16,f32>(&c,op,xs,ints),9=>nll_dispatch::<f64,f64>(&c,op,xs,ints),_=>panic!("NLL requires real floating input")}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
