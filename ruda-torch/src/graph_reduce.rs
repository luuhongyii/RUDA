use super::*;
use super::graph_contract::{NodeSpec, NO_WEIGHT, RMS_NORM};

#[ruda(launch)]
fn suffix_reduce<F: Float + RudaElement, O: Float + RudaElement, V: Size>(input: &Array<Vector<F,V>>, output: &mut Array<O>,
    width: usize, factor: f32, #[comptime] mean: bool, #[comptime] multiply: bool,
    #[comptime] threads: usize, #[define(V)] vector: usize,
) {
    let mut row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize % threads;
    if comptime!(threads == 32) { row = row * 4usize + UNIT_POS as usize / 32usize; }
    if row < output.len() {
        let mut local = Vector::<f32,V>::new(0.0f32);
        let mut col = lane;
        while col < width / vector {
            local += Vector::<f32,V>::cast_from(input[row * (width/vector) + col]);
            col += threads;
        }
        let mut scalar = 0.0f32;
        #[unroll] for element in 0usize..vector { scalar += local[element]; }
        let mut sum = plane_sum(scalar);
        if comptime!(threads > 32) {
            let mut partials = SharedMemory::<f32>::new(threads/32);
            if lane % 32 == 0 { partials[lane/32] = sum; }
            sync_ruda();
            let mut value = 0.0f32;
            if lane % 32 < threads/32 { value = partials[lane%32]; }
            sum = plane_sum(value);
        }
        if comptime!(mean) {
            if comptime!(multiply) { sum *= factor; } else { sum /= width as f32; }
        }
        if lane == 0 { output[row] = O::cast_from(sum); }
    }
}

#[ruda(launch)]
fn column_reduce<F: Float + RudaElement, O: Float + RudaElement>(input: &Array<F>, output: &mut Array<O>,
    factor: f32, #[comptime] width: usize, #[comptime] inner: usize,
    #[comptime] mean: bool, #[comptime] multiply: bool,
) {
    let tiles = comptime!(inner.div_ceil(32));
    let outer = RUDA_POS_X as usize / tiles;
    let lane = UNIT_POS_X as usize;
    let part = UNIT_POS_Y as usize;
    let column = (RUDA_POS_X as usize % tiles) * 32usize + lane;
    let mut sum = 0.0f32;
    if column < inner {
        let mut row = part;
        while row < width {
            sum += f32::cast_from(input[(outer * width + row) * inner + column]);
            row += 8usize;
        }
    }
    let mut partials = SharedMemory::<f32>::new(256usize);
    partials[part*32+lane] = sum;
    sync_ruda();
    if part == 0 && column < inner {
        let mut value = partials[lane];
        #[unroll] for p in 1usize..8 { value += partials[p*32+lane]; }
        if comptime!(mean) {
            if comptime!(multiply) { value *= factor; } else { value /= width as f32; }
        }
        output[outer*inner+column] = O::cast_from(value);
    }
}

pub(super) fn contiguous_axes(shape: &[usize], mask: u32) -> Option<(usize,usize)> {
    if shape.len() > 32 { return None; }
    let first = (0..shape.len()).find(|&d| mask & (1 << d) != 0)?;
    let last = (0..shape.len()).rfind(|&d| mask & (1 << d) != 0)?;
    if (first..=last).any(|d|mask & (1 << d) == 0 && shape[d] != 1) { return None; }
    Some((shape[..first].iter().product(),shape[last+1..].iter().product()))
}

pub(super) fn prepare_sum(c: &ComputeClient<CudaRuntime>, a: &View, out: &View,
    mask: u32, mean: bool, multiply: bool) -> Option<PreparedKernel<CudaRuntime>>
{
    if a.dtype > 2 || out.dtype > 2 || !(a.dtype == out.dtype || out.dtype == 0)
        || a.len == 0 || a.len > u32::MAX as usize || out.len == 0
        || !a.contiguous() || !out.contiguous() { return None; }
    let (outer, inner) = contiguous_axes(&a.shape, mask)?;
    if outer.checked_mul(inner)? != out.len { return None; }
    let width = a.len / out.len;
    if width == 0 || width > 65536 || inner != 1 && (inner < 16 || width < 8) { return None; }
    let rows = out.len;
    let threads = if width <= 128 || width <= 1024 && rows >= 128 {32}
        else if width <= 4096 {128} else {256};
    let vector = super::row_cached::memory_vector_size(c,width,&[a],16);
    let blocks = if inner == 1 {if threads == 32 {rows.div_ceil(4)} else {rows}}
        else {outer.checked_mul(inner.div_ceil(32))?};
    if blocks > c.properties().hardware.max_ruda_count.0 as usize { return None; }
    let grid = RudaCount::Static(u32::try_from(blocks).ok()?,1,1);
    let factor = rows as f32 / a.len as f32;
    macro_rules! run { ($s:ty, $o:ty) => { unsafe {
        if inner == 1 {
            suffix_reduce::prepare::<$s,$o,CudaRuntime>(c,grid,RudaDim::new_1d(threads.max(128) as u32),
                a.array_arg(),out.array_arg(),width,factor,mean,multiply,threads,vector)
        } else {
            column_reduce::prepare::<$s,$o,CudaRuntime>(c,grid,RudaDim::new_2d(32,8),
                a.array_arg(),out.array_arg(),factor,width,inner,mean,multiply)
        }
    }}; }
    Some(match (a.dtype,out.dtype) {
        (0,0)=>run!(f32,f32), (1,1)=>run!(f16,f16), (2,2)=>run!(bf16,bf16),
        (1,0)=>run!(f16,f32), (2,0)=>run!(bf16,f32), _=>return None,
    })
}

#[ruda(launch)]
fn row_reduce<F: Float + RudaElement>(
    input: &Array<F>, output: &mut Array<F>, width: usize,
    #[comptime] shape: Vec<usize>, #[comptime] strides: Vec<usize>,
    #[comptime] mask: u32, #[comptime] mean: bool,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut remaining = row;
    let mut base = 0usize;
    #[unroll]
    for reverse in 0usize..shape.len() {
        let dim = comptime!(shape.len()-1-reverse);
        if comptime!(mask & (1u32 << dim) == 0) {
            base += (remaining % comptime!(shape[dim])) * comptime!(strides[dim]);
            remaining /= comptime!(shape[dim]);
        }
    }
    let mut sum = 0.0f32;
    let mut column = lane;
    while column < width {
        let mut index = column;
        let mut address = base;
        #[unroll]
        for reverse in 0usize..shape.len() {
            let dim = comptime!(shape.len()-1-reverse);
            if comptime!(mask & (1u32 << dim) != 0) {
                address += (index % comptime!(shape[dim])) * comptime!(strides[dim]);
                index /= comptime!(shape[dim]);
            }
        }
        sum += f32::cast_from(input[address]);
        column += 256usize;
    }
    sum = plane_sum(sum);
    let mut partials = SharedMemory::<f32>::new(8usize);
    if lane % 32usize == 0 { partials[lane / 32usize] = sum; }
    sync_ruda();
    if lane < 32usize {
        let mut value = 0.0f32;
        if lane < 8usize { value = partials[lane]; }
        value = plane_sum(value);
        if comptime!(mean) { value /= width as f32; }
        if lane == 0 { output[row] = F::cast_from(value); }
    }
}

pub(super) fn prepare(c: &ComputeClient<CudaRuntime>, tensors: &[View], n: &NodeSpec,
    output: usize) -> Option<PreparedKernel<CudaRuntime>>
{
    if !matches!(n.op, RMS_NORM | 102..=107 | 119 | 120) { return None; }
    let a = &tensors[n.a as usize]; let out = &tensors[output];
    if a.dtype > 2 || a.dtype != out.dtype || a.len == 0 || out.len == 0 || !a.contiguous() || !out.contiguous() { return None; }
    let b = if n.b == NO_WEIGHT { a } else { &tensors[n.b as usize] };
    let reduction = matches!(n.op, 104 | 105 | 119 | 120);
    let mask = if reduction { n.scalar as u32 } else { 0 };
    let width = if reduction {
        a.shape.iter().enumerate().filter(|(d,_)| mask & (1 << d) != 0).map(|(_,d)|d).product()
    } else { *a.shape.last()? };
    let rows = a.len / width;
    if reduction {
        if let Some(kernel) = prepare_sum(c,a,out,mask,matches!(n.op,105|120),false) { return Some(kernel); }
        if width < 32 { return None; }
    } else if n.op == RMS_NORM {
        if !super::row_cached::eligible(width) { return None; }
    } else if n.scalar as usize + 1 != a.shape.len() || !(32..=32768).contains(&width) || !b.contiguous() {
        return None;
    }
    macro_rules! run { ($ty:ty) => { unsafe {
        if reduction {
            row_reduce::prepare::<$ty,CudaRuntime>(c,
                RudaCount::Static(u32::try_from(rows).ok()?,1,1),RudaDim::new_1d(256),
                a.array_arg(),out.array_arg(),width,a.shape.clone(),a.strides.clone(),mask,matches!(n.op,105|120))
        } else if n.op == RMS_NORM {
            let threads = super::row_cached::norm_threads(width,rows);
            let vector = super::row_cached::memory_vector_size(c,width,&[a,out],16);
            super::row_cached::norm::prepare::<$ty,$ty,$ty,$ty,CudaRuntime>(c,
                super::row_cached::norm_grid(rows,threads),super::row_cached::norm_block(threads),
                a.arg(),b.arg(),a.arg(),out.arg(),out.arg(),out.arg(),n.scalar,
                n.b!=NO_WEIGHT,false,false,false,false,width,threads,vector)
        } else {
            let threads = if width <= 1024 { if rows>=1024 || width<512 {32} else {128} }
                else if width<=16384 {super::row_cached::threads(width)} else {1024};
            let vector = super::row_cached::memory_vector_size(c,width,&[a,b,out],16);
            let blocks = if threads==32 {rows.div_ceil(4)} else {rows};
            let grid=RudaCount::Static(u32::try_from(blocks).ok()?,1,1);
            let dim=RudaDim::new_1d(threads.max(128) as u32);
            if n.op>=106 {
                super::row_cached::softmax_backward::prepare::<$ty,$ty,CudaRuntime>(c,grid,dim,
                    a.arg(),b.arg(),out.arg(),n.op==107,width,threads,vector)
            } else {
                super::row_cached::softmax::prepare::<$ty,$ty,CudaRuntime>(c,grid,dim,
                    a.arg(),out.arg(),n.op==103,width,threads,vector)
            }
        }
    }}; }
    Some(match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()})
}
