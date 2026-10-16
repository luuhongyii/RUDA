use super::*;

#[ruda(launch)]
fn moments_tile<E: Numeric + RudaElement, F: Float + RudaElement>(
    input: &Tensor<E>, input_m2: &Tensor<F>, input_counts: &Tensor<i64>,
    means: &mut Tensor<F>, m2: &mut Tensor<F>, counts: &mut Tensor<i64>,
    length: usize, blocks: usize, #[comptime] first: bool)
{
    let task = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let row = task / blocks;
    let column = (task % blocks) * 256 + lane;
    let mut shared_mean = SharedMemory::<F>::new(256usize);
    let mut shared_m2 = SharedMemory::<F>::new(256usize);
    let mut shared_count = SharedMemory::<u32>::new(256usize);
    let mut mean = F::cast_from(0);
    let mut moment = F::cast_from(0);
    let mut count = 0u32;
    if column < length {
        let offset = row * length + column;
        mean = F::cast_from(input[offset]);
        if comptime!(first) { moment = mean - mean; count = 1; }
        else { moment = input_m2[offset]; count = u32::cast_from(input_counts[offset]); }
    }
    shared_mean[lane] = mean;
    shared_m2[lane] = moment;
    shared_count[lane] = count;
    sync_ruda();
    let mut stride = 128usize;
    while stride > 0 {
        if lane < stride {
            let na = shared_count[lane];
            let nb = shared_count[lane + stride];
            if nb > 0 {
                if na == 0 {
                    shared_mean[lane] = shared_mean[lane + stride];
                    shared_m2[lane] = shared_m2[lane + stride];
                } else {
                    let a = shared_mean[lane];
                    let delta = shared_mean[lane + stride] - a;
                    let weight = F::cast_from(nb) / F::cast_from(na + nb);
                    shared_mean[lane] = a + delta * weight;
                    shared_m2[lane] = shared_m2[lane] + shared_m2[lane + stride]
                        + delta * delta * F::cast_from(na) * weight;
                }
                shared_count[lane] = na + nb;
            }
        }
        sync_ruda();
        stride /= 2;
    }
    if lane == 0 {
        means[task] = shared_mean[0];
        m2[task] = shared_m2[0];
        counts[task] = i64::cast_from(shared_count[0]);
    }
}

fn allocate(c: &ComputeClient<CudaRuntime>, rows: usize, blocks: usize, dtype: u32) -> View {
    let len = rows.checked_mul(blocks).expect("statistics workspace overflow");
    View::packed(c.empty(len.max(1).checked_mul(element_bytes(dtype)).expect("statistics byte overflow")),
        vec![rows, blocks], dtype)
}

fn stages<E: Numeric + RudaElement, F: Float + RudaElement>(c: &ComputeClient<CudaRuntime>, vs: &[View])
    -> (Vec<PreparedKernel<CudaRuntime>>, Vec<View>)
{
    let rows = vs[0].shape[0];
    let mut length = vs[0].shape[1];
    let mut nodes = Vec::new();
    let mut scratch: Vec<View> = Vec::new();
    if rows == 0 { return (nodes, scratch); }
    let mut previous: Option<usize> = None;
    loop {
        let blocks = length.div_ceil(256);
        let start = scratch.len();
        if blocks > 1 {
            scratch.push(allocate(c, rows, blocks, vs[1].dtype));
            scratch.push(allocate(c, rows, blocks, vs[2].dtype));
        }
        scratch.push(allocate(c, rows, blocks, 4));
        let mean = if blocks == 1 { &vs[1] } else { &scratch[start] };
        let moment = if blocks == 1 { &vs[2] } else { &scratch[start + 1] };
        let counts = scratch.last().unwrap();
        let grid = RudaCount::Static(u32::try_from(rows.checked_mul(blocks).unwrap()).expect("statistics grid overflow"), 1, 1);
        let node = if let Some(index) = previous {
            moments_tile::prepare::<F, F, CudaRuntime>(c, grid, RudaDim::new_1d(256),
                scratch[index].arg(), scratch[index + 1].arg(), scratch[index + 2].arg(),
                mean.arg(), moment.arg(), counts.arg(), length, blocks, false)
        } else {
            moments_tile::prepare::<E, F, CudaRuntime>(c, grid, RudaDim::new_1d(256),
                vs[0].arg(), moment.arg(), counts.arg(), mean.arg(), moment.arg(), counts.arg(), length, blocks, true)
        };
        nodes.push(node);
        if blocks == 1 { break; }
        previous = Some(start);
        length = blocks;
    }
    (nodes, scratch)
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_statistics_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_statistics(descriptors: *const Descriptor, count: usize) -> i32 {
    checked(|| {
        assert_eq!(count, 3);
        let vs: Vec<_> = unsafe { abi::slice(descriptors, count) }.iter().map(|d| unsafe { abi::view(d) }).collect();
        for v in &vs { operation_graph::packed(v); assert!(v.len <= u32::MAX as usize); }
        assert_eq!(vs[0].shape.len(), 2);
        assert!(vs[0].shape[1] > 0 && vs[0].shape[1] <= u32::MAX as usize);
        let dtype = if vs[0].dtype == 9 { 9 } else { 0 };
        for out in &vs[1..] {
            assert_eq!(out.shape, vec![vs[0].shape[0]]);
            assert_eq!(out.dtype, dtype);
        }
        let c = client().fixed_execution_queue();
        let (nodes, _workspace) = match vs[0].dtype {
            0 => stages::<f32, f32>(&c, &vs), 1 => stages::<f16, f32>(&c, &vs),
            2 => stages::<bf16, f32>(&c, &vs), 9 => stages::<f64, f64>(&c, &vs),
            _ => panic!("moments require real floating input"),
        };
        let launches = nodes.len();
        for node in nodes {
            let (task, grid, arguments, queue) = node.into_parts();
            queue.launch(task, grid, arguments);
        }
        LAUNCHES.fetch_add(launches as u64, Ordering::Relaxed);
        finish_dispatch(&c);
    })
}
