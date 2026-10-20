use super::*;

#[ruda(launch)]
fn scan_step<E: Numeric + RudaElement>(input: &Tensor<E>, keys: &Tensor<i64>,
    output: &mut Tensor<E>, indices: &mut Tensor<i64>, length: usize, step: usize,
    #[comptime] maximum: bool, #[comptime] first: bool)
{
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let column = p % length;
        let mut value = input[p];
        let mut index = i64::cast_from(column);
        if comptime!(!first) { index = keys[p]; }
        if column >= step {
            let left = input[p - step];
            let mut choose = left > value;
            if comptime!(!maximum) { choose = left < value; }
            choose = choose || (left != left && value == value);
            if choose {
                value = left;
                if comptime!(first) { index = i64::cast_from(column - step); }
                else { index = keys[p - step]; }
            }
        }
        output[p] = value;
        indices[p] = index;
    }
}

#[ruda(launch)]
fn search<E: Numeric + RudaElement, I: Numeric + RudaElement>(sequence: &Tensor<E>,
    queries: &Tensor<E>, output: &mut Tensor<I>, #[comptime] right: bool)
{
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let length = sequence.shape(1);
        let base = (p / queries.shape(1)) * length;
        let value = queries[p];
        let mut low = 0usize;
        let mut high = length;
        while low < high {
            let middle = low + (high - low) / 2;
            let boundary = sequence[base + middle];
            let mut advance = !(boundary >= value);
            if comptime!(right) { advance = !(boundary > value); }
            if advance { low = middle + 1; } else { high = middle; }
        }
        output[p] = I::cast_from(low);
    }
}

fn allocate(c: &ComputeClient<CudaRuntime>, shape: &[usize], dtype: u32) -> View {
    let len = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("ordering workspace overflow");
    View::packed(c.empty(len.max(1).checked_mul(element_bytes(dtype)).expect("ordering workspace bytes overflow")), shape.to_vec(), dtype)
}

fn scan<E: Numeric + RudaElement>(c: &ComputeClient<CudaRuntime>, vs: &[View], maximum: bool)
    -> (Vec<PreparedKernel<CudaRuntime>>, Vec<View>)
{
    let input = &vs[0];
    let length = input.shape[1];
    let mut scratch = Vec::new();
    if length > 2 {
        scratch.push(allocate(c, &input.shape, input.dtype));
        scratch.push(allocate(c, &input.shape, 4));
    }
    if length > 4 {
        scratch.push(allocate(c, &input.shape, input.dtype));
        scratch.push(allocate(c, &input.shape, 4));
    }
    let count = RudaCount::Static(u32::try_from(input.len.div_ceil(128)).unwrap(), 1, 1);
    let mut nodes = Vec::new();
    let mut step = 1;
    let mut slot = 0;
    loop {
        let last = step >= length.div_ceil(2);
        let source = if step == 1 { input } else { &scratch[2 - slot] };
        let keys = if step == 1 { &vs[2] } else { &scratch[3 - slot] };
        let out = if last { &vs[1] } else { &scratch[slot] };
        let indices = if last { &vs[2] } else { &scratch[slot + 1] };
        nodes.push(scan_step::prepare::<E, CudaRuntime>(c, count.clone(), RudaDim::new_1d(128),
            source.arg(), keys.arg(), out.arg(), indices.arg(), length, step, maximum, step == 1));
        if last { break; }
        slot = 2 - slot;
        step *= 2;
    }
    (nodes, scratch)
}

fn validate(kind: u32, vs: &[View], ints: &[i64]) {
    assert!(kind <= 2);
    for v in vs { operation_graph::packed(v); }
    if kind == 0 { operation_reduce::validate(vs, ints, &[]); return; }
    assert_eq!(vs.len(), 3);
    assert_eq!(ints.len(), 1);
    assert!(matches!(ints[0], 0 | 1));
    assert_eq!(vs[0].shape.len(), 2);
    assert!(vs[0].dtype <= 9);
    if kind == 1 {
        assert_eq!(vs[1].shape, vs[0].shape);
        assert_eq!(vs[2].shape, vs[0].shape);
        assert_eq!(vs[1].dtype, vs[0].dtype);
        assert_eq!(vs[2].dtype, 4);
    } else {
        assert_ne!(vs[0].dtype, 3, "searchsorted does not support bool");
        assert_eq!(vs[1].shape.len(), 2);
        assert_eq!(vs[1].shape[0], vs[0].shape[0]);
        assert_eq!(vs[1].dtype, vs[0].dtype);
        assert_eq!(vs[2].shape, vs[1].shape);
        assert!(matches!(vs[2].dtype, 4 | 5));
        if vs[2].dtype == 5 { assert!(vs[0].shape[1] < i32::MAX as usize + 1); }
    }
}

fn prepare(c: &ComputeClient<CudaRuntime>, kind: u32, vs: &[View], ints: &[i64])
    -> (Vec<PreparedKernel<CudaRuntime>>, Vec<View>)
{
    if kind == 0 { return operation_reduce::prepare(c, vs, ints); }
    if vs[2].len == 0 { return (Vec::new(), Vec::new()); }
    let count = RudaCount::Static(u32::try_from(vs[2].len.div_ceil(128)).unwrap(), 1, 1);
    macro_rules! run {
        ($e:ty) => {
            if kind == 1 { scan::<$e>(c, vs, ints[0] != 0) } else {
                let node = if vs[2].dtype == 4 {
                    search::prepare::<$e, i64, CudaRuntime>(c, count, RudaDim::new_1d(128), vs[0].arg(), vs[1].arg(), vs[2].arg(), ints[0] != 0)
                } else {
                    search::prepare::<$e, i32, CudaRuntime>(c, count, RudaDim::new_1d(128), vs[0].arg(), vs[1].arg(), vs[2].arg(), ints[0] != 0)
                };
                (vec![node], Vec::new())
            }
        };
    }
    match vs[0].dtype {
        0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), 3 | 8 => run!(u8),
        4 => run!(i64), 5 => run!(i32), 6 => run!(i16), 7 => run!(i8), 9 => run!(f64),
        _ => unreachable!(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_ordering_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_ordering(kind: u32, descriptors: *const Descriptor,
    count: usize, integers: *const i64, integer_count: usize) -> i32
{
    checked(|| {
        assert!(kind <= 2 && (2..=3).contains(&count) && integer_count <= 3);
        let vs: Vec<_> = unsafe { abi::slice(descriptors, count) }.iter().map(|d| unsafe { abi::view(d) }).collect();
        let ints = unsafe { abi::slice(integers, integer_count) };
        validate(kind, &vs, ints);
        let c = client().fixed_execution_queue();
        let (nodes, _workspace) = prepare(&c, kind, &vs, ints);
        let launches = nodes.len();
        for node in nodes {
            let (task, count, arguments, queue) = node.into_parts();
            queue.launch(task, count, arguments);
        }
        LAUNCHES.fetch_add(launches as u64, Ordering::Relaxed);
        finish_dispatch(&c);
    })
}
