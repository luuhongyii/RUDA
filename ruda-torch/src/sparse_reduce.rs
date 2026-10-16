use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda]
fn compare(input: &Tensor<i64>, row: usize, grad: &Tensor<i64>, candidate: usize,
    #[comptime] kept: Vec<i64>,
) -> i32 {
    let mut result = 0i32;
    #[unroll]
    for d in 0..comptime!(kept.len()) {
        if result == 0i32 {
            let axis = comptime!(kept[d] as usize);
            let x = input[axis * input.stride(0) + row * input.stride(1)];
            let y = grad[d * grad.stride(0) + candidate * grad.stride(1)];
            if x < y { result = -1i32; } else if x > y { result = 1i32; }
        }
    }
    result
}

#[ruda(launch)]
fn match_rows(input: &Tensor<i64>, grad: &Tensor<i64>, matched: &mut Tensor<u32>,
    #[comptime] kept: Vec<i64>,
) {
    let row = position();
    if row < matched.len() {
        let mut lo = 0usize; let mut hi = grad.shape(1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if compare(input, row, grad, mid, kept.clone()) > 0i32 { lo = mid + 1usize; } else { hi = mid; }
        }
        let mut found = 4294967295u32;
        if lo < grad.shape(1) {
            if compare(input, row, grad, lo, kept.clone()) == 0i32 { found = u32::cast_from(lo); }
        }
        matched[row] = found;
    }
}

#[ruda(launch)]
fn gather_blocks(grad: &Tensor<u8>, matched: &Tensor<u32>, output: &mut Tensor<u8>, width: usize) {
    let p = position();
    if p < output.len() {
        let row = p / width; let source = matched[row]; let mut value = 0u8;
        if source != 4294967295u32 { value = grad[address(grad, source as usize * width + p % width)]; }
        output[address(output, p)] = value;
    }
}

fn grid(work: usize) -> RudaCount {
    let blocks = work.div_ceil(256); assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

pub(super) fn dispatch(xs: &[View], kept: &[i64]) {
    assert_eq!(xs.len(),4);
    let (input,grad,values,out)=(&xs[0],&xs[1],&xs[2],&xs[3]);
    for indices in [input,grad] {assert_eq!(indices.dtype,4);assert_eq!(indices.shape.len(),2);}
    assert!(!kept.is_empty()); assert_eq!(kept.len(),grad.shape[0]);
    assert!(kept.iter().all(|&axis|axis>=0 && (axis as usize)<input.shape[0]));
    assert!(kept.windows(2).all(|pair|pair[0]<pair[1]));
    assert!(!values.shape.is_empty()); assert!(!out.shape.is_empty());
    assert_eq!(values.shape[0],grad.shape[1]); assert_eq!(out.shape[0],input.shape[1]);
    assert_eq!(values.shape[1..],out.shape[1..]); assert_eq!(values.dtype,out.dtype);
    if out.len==0 {return;}
    let c=client(); let block=RudaDim::new_1d(256);
    let matched=View::packed(c.empty(input.shape[1].checked_mul(4).expect("sparse sum gradient map overflow")),vec![input.shape[1]],5);
    let source_bytes=values.byte_view(); let output_bytes=out.byte_view();
    scientific::indexable(&matched); scientific::indexable(&source_bytes); scientific::indexable(&output_bytes);
    let width=output_bytes.shape[1..].iter().try_fold(1usize,|n,&s|n.checked_mul(s)).expect("sparse sum gradient block overflow");
    unsafe {
        match_rows::launch::<CudaRuntime>(&c,grid(matched.len),block,input.arg(),grad.arg(),matched.arg(),kept.to_vec());
        gather_blocks::launch::<CudaRuntime>(&c,grid(output_bytes.len),block,source_bytes.arg(),matched.arg(),output_bytes.arg(),width);
    }
    LAUNCHES.fetch_add(2,Ordering::Relaxed); finish_dispatch(&c);
}
