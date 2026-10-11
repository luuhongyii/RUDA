pub type TestRuntime = crate::CpuRuntime;

pub use half::f16;

#[ruda_kernel::dsl::prelude::ruda]
fn typed_pointer_read(
    pointer: ruda_kernel::dsl::prelude::Ptr<f32, ruda_kernel::dsl::prelude::GlobalSpace>,
) -> f32 {
    pointer.load()
}

#[ruda_kernel::dsl::prelude::ruda(launch)]
fn typed_pointer_kernel(
    input: &ruda_kernel::dsl::prelude::Array<f32>,
    output: &mut ruda_kernel::dsl::prelude::Array<f32>,
) {
    let index = UNIT_POS as usize;
    let input_view = input
        .to_slice()
        .typed::<ruda_kernel::dsl::prelude::GlobalSpace>();
    let source = input_view
        .ptr(0)
        .offset(index)
        .select(index % 2 == 0, input_view.ptr(0));
    let value = typed_pointer_read(source);
    let mut local = ruda_kernel::dsl::prelude::Array::<f32>::new(1usize);
    let local_ptr = local
        .to_slice_mut()
        .typed::<ruda_kernel::dsl::prelude::Local>()
        .ptr(0);
    local_ptr.store(value);
    let mut shared = ruda_kernel::dsl::prelude::SharedMemory::<f32>::new(4usize);
    let shared_ptr = shared
        .to_slice_mut()
        .typed::<ruda_kernel::dsl::prelude::Workgroup>()
        .ptr(index);
    shared_ptr.store(input_view.ptr(index).load());
    ruda_kernel::dsl::prelude::sync_ruda();
    let result = value + local_ptr.read_only().load() + shared_ptr.load();
    output
        .to_slice_mut()
        .typed::<ruda_kernel::dsl::prelude::GlobalSpace>()
        .ptr(index)
        .store(result);
}

#[test]
fn test_typed_pointer_spaces_permissions_offsets_and_selection_cpu() {
    use ruda_kernel::dsl::prelude::*;
    let client = TestRuntime::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(4 * core::mem::size_of::<f32>());
    unsafe {
        typed_pointer_kernel::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(input, 4),
            ArrayArg::from_raw_parts(output.clone(), 4),
        );
    }
    let bytes = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&bytes), &[3.0, 4.0, 9.0, 6.0]);
}

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

ruda_kernel::dsl::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
ruda_kernel::library::testgen!();
ruda_kernel::library::testgen_tensor_identity!([f16, f32, u32]);
ruda_kernel::library::testgen_quantized_view!(f32);

#[ruda(launch)]
fn barrier_smoke(out: &mut Array<f32>) {
    let barrier = barrier::Barrier::local();
    barrier.arrive_and_wait();
    if UNIT_POS == 0 {
        out[0] = 1.0;
    }
}

#[ruda(launch)]
fn sync_ruda_magic(out: &mut Array<u32>) {
    let mut mem = SharedMemory::<u32>::new(1usize);
    if UNIT_POS == 0 {
        mem[0] = 0xDEADBEEFu32;
    }
    sync_ruda();
    out[UNIT_POS as usize] = mem[0];
}

#[ruda(launch)]
fn sync_ruda_two_phase(out: &mut Array<u32>) {
    let mut mem = SharedMemory::<u32>::new(4usize);
    let idx = UNIT_POS as usize;
    mem[idx] = (idx as u32) + 1;
    sync_ruda();

    if UNIT_POS == 0 {
        let mut sum = 0u32;
        for i in 0..4 {
            sum += mem[i];
        }
        mem[0] = sum;
    }
    sync_ruda();

    out[idx] = mem[0];
}

#[ruda(launch)]
fn sync_ruda_all_reduce(out: &mut Array<u32>) {
    let mut mem = SharedMemory::<u32>::new(8usize);
    let idx = UNIT_POS as usize;
    mem[idx] = idx as u32;
    sync_ruda();

    let mut sum = 0u32;
    for i in 0..8 {
        sum += mem[i];
    }
    out[idx] = sum;
}

#[test]
fn test_barrier_smoke_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(core::mem::size_of::<f32>());

    unsafe {
        barrier_smoke::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(out.clone(), 1),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&bytes);
    assert_eq!(actual[0], 1.0);
}

#[test]
fn test_sync_ruda_magic_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(4 * core::mem::size_of::<u32>());

    unsafe {
        sync_ruda_magic::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(out.clone(), 4),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[0xDEADBEEF; 4]);
}

#[test]
fn test_sync_ruda_two_phase_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(4 * core::mem::size_of::<u32>());

    unsafe {
        sync_ruda_two_phase::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(out.clone(), 4),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[10u32; 4]);
}

#[test]
fn test_sync_ruda_all_reduce_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(8 * core::mem::size_of::<u32>());

    unsafe {
        sync_ruda_all_reduce::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(8),
            ArrayArg::from_raw_parts(out.clone(), 8),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[28u32; 8]);
}
