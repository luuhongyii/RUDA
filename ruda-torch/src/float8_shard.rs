use super::*;
use ruda_kernel::quantization::float8_block as kernels;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_float8_shard_api_version() -> u32 {1}

pub(super) fn validate(vs: &[View], encode: bool, ints: &[i64]) {
    assert_eq!((vs.len(), ints.len()), if encode {(4, 8)} else {(2, 6)});
    assert!(ints.iter().all(|&n| n >= 0 && n <= u32::MAX as i64));
    for v in vs {
        operation_graph::packed(v);
        assert!(v.shape.iter().all(|&n| n <= u32::MAX as usize));
    }
    let (br, bc, row, col, rows, cols) =
        (ints[0] as usize, ints[1] as usize, ints[2] as usize, ints[3] as usize, ints[4] as usize, ints[5] as usize);
    assert!(br > 0 && bc > 0 && bc % 32 == 0);
    assert!(br.checked_mul(bc).is_some_and(|n| n <= u32::MAX as usize - 32));
    assert!(rows.div_ceil(br).checked_mul(br).is_some_and(|n| n <= u32::MAX as usize));
    assert!(cols.div_ceil(bc).checked_mul(bc).is_some_and(|n| n <= u32::MAX as usize));
    assert_eq!(vs[0].shape.len(), 2);
    assert!(vs[0].dtype <= 2 && row <= rows && col <= cols);
    assert!(vs[0].shape[0] <= rows - row && vs[0].shape[1] <= cols - col);
    assert_eq!(vs[1].dtype, 0);
    assert_eq!(vs[1].shape, vec![2, rows.div_ceil(br), cols.div_ceil(bc)]);
    if encode {
        assert!(matches!(vs[2].dtype, 12 | 13) && ints[6] <= 16 && ints[7] <= 1);
        let padded = [vs[0].shape[0] + row % br, vs[0].shape[1] + col % bc];
        assert_eq!(vs[2].shape.as_slice(), &padded);
        assert_eq!(vs[3].dtype, 0);
        assert_eq!(vs[3].shape, vec![padded[0].div_ceil(br), padded[1].div_ceil(bc)]);
    }
}

pub(super) fn prepare(c: &ComputeClient<CudaRuntime>, vs: &[View], encode: bool,
    ints: &[i64]) -> Vec<PreparedKernel<CudaRuntime>> {
    validate(vs, encode, ints);
    let ts: Vec<_> = vs.iter().map(|v| {
        let mut t = primitives::tensor(v); t.client = c.clone(); t
    }).collect();
    let blocks = if encode {vs[3].len} else {vs[1].len / 2};
    assert!(blocks <= c.properties().hardware.max_ruda_count.0 as usize);
    if blocks == 0 {return Vec::new();}
    let grid = RudaCount::Static(blocks as u32, 1, 1);
    let (rows, cols) = (vs[0].shape[0] as u32, vs[0].shape[1] as u32);
    let (br, bc, row, col) = (ints[0] as u32, ints[1] as u32, ints[2] as u32, ints[3] as u32);
    let (groups, global_blocks) = (vs[1].shape[2] as u32, (vs[1].len / 2) as u32);
    vec![unsafe {if encode {
        let maximum = if vs[2].dtype == 12 {448.0f32} else {57344.0f32};
        kernels::quantize_shard::prepare::<CudaRuntime>(c, grid, RudaDim::new_1d(32),
            ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(),
            ts[2].clone().into_array_arg(), ts[3].clone().into_array_arg(),
            rows, cols, row, col, groups, global_blocks, vs[2].shape[0] as u32, vs[2].shape[1] as u32,
            vs[3].shape[1] as u32, br, bc, maximum * 2.0f32.powi(-(ints[6] as i32)),
            if vs[2].dtype == 12 {18} else {19}, ints[7] != 0, ts[0].dtype.into())
    } else {
        kernels::shard_statistics::prepare::<CudaRuntime>(c, grid, RudaDim::new_1d(32),
            ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(),
            rows, cols, row, col, groups, global_blocks, br, bc, ts[0].dtype.into())
    }}]
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_float8_shard(op: u32, descriptors: *const Descriptor, count: usize,
    integers: *const i64, integer_count: usize) -> i32 {
    checked(|| {
        assert!(op <= 1 && count == if op == 0 {2} else {4});
        assert_eq!(integer_count, if op == 0 {6} else {8});
        let vs: Vec<_> = unsafe {abi::slice(descriptors, count)}.iter().map(|d| unsafe {abi::view(d)}).collect();
        let ints = unsafe {abi::slice(integers, integer_count)};
        validate(&vs, op == 1, ints);
        let c = client();
        let blocks = if op == 1 {vs[3].len} else {vs[1].len / 2};
        assert!(blocks <= c.properties().hardware.max_ruda_count.0 as usize);
        if blocks != 0 {
            let ts: Vec<_> = vs.iter().map(primitives::tensor).collect();
            let grid = RudaCount::Static(blocks as u32, 1, 1);
            let (rows, cols) = (vs[0].shape[0] as u32, vs[0].shape[1] as u32);
            let (br, bc, row, col) = (ints[0] as u32, ints[1] as u32, ints[2] as u32, ints[3] as u32);
            let (groups, global_blocks) = (vs[1].shape[2] as u32, (vs[1].len / 2) as u32);
            if op == 1 {
                let maximum = if vs[2].dtype == 12 {448.0f32} else {57344.0f32};
                unsafe {kernels::quantize_shard::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(32),
                    ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(),
                    ts[2].clone().into_array_arg(), ts[3].clone().into_array_arg(),
                    rows, cols, row, col, groups, global_blocks, vs[2].shape[0] as u32, vs[2].shape[1] as u32,
                    vs[3].shape[1] as u32, br, bc, maximum * 2.0f32.powi(-(ints[6] as i32)),
                    if vs[2].dtype == 12 {18} else {19}, ints[7] != 0, ts[0].dtype.into());}
            } else {
                unsafe {kernels::shard_statistics::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(32),
                    ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(),
                    rows, cols, row, col, groups, global_blocks, br, bc, ts[0].dtype.into());}
            }
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        }
        finish_dispatch(&c);
    })
}
