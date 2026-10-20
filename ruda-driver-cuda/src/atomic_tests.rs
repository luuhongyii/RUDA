use crate::{CudaDevice, CudaRuntime};
use half::{bf16, f16};
use ruda_core::ir::features::AtomicUsage;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn contended_add<F: Numeric, N: Size>(
    state: Array<Atomic<Vector<F, N>>>,
    old: &mut Array<Vector<F, N>>,
) {
    old[ABSOLUTE_POS] = state[0].fetch_add(Vector::from_int(1));
}

#[ruda(launch)]
fn shared_atomic_add<F: Float, N: Size>(out: &mut Array<Vector<F, N>>, #[comptime] native: bool) {
    let mut state = SharedMemory::<Atomic<Vector<F, N>>>::new(1usize);
    let pointer = state.to_slice_mut().typed::<Workgroup>().ptr(0).value();
    let imported = ValuePtr::<Atomic<Vector<F, N>>, Workgroup, ReadWrite>::from_native(
        pointer.native_address(),
    );
    if UNIT_POS == 0 {
        state[0].store(Vector::from_int(0));
    }
    sync_ruda();
    if native {
        out[UNIT_POS as usize] = imported.fetch_add(Vector::from_int(1));
    } else {
        out[UNIT_POS as usize] = state[0].fetch_add(Vector::from_int(1));
    }
    sync_ruda();
    if UNIT_POS == 0 {
        out[64] = imported.atomic_load();
        out[65] = imported.fetch_sub(Vector::from_int(1));
        out[66] = imported.atomic_load();
        out[67] = imported.compare_exchange_weak(Vector::from_int(63), Vector::from_int(71));
        out[68] = imported.atomic_load();
    }
}

#[ruda(launch)]
fn bit_roundtrip<F: Numeric, N: Size>(
    state: Array<Atomic<Vector<F, N>>>,
    values: Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
) {
    out[0] = state[0].load();
    out[1] = state[0].load();
    state[0].store(values[0]);
    out[2] = state[0].load();
    out[3] = state[0].swap(values[1]);
    out[4] = state[0].load();
    out[5] = state[0].compare_exchange_weak(values[2], values[3]);
    out[6] = state[0].load();
    out[7] = state[0].compare_exchange_weak(values[1], values[4]);
    out[8] = state[0].load();
    out[9] = state[1].load();
}

#[ruda(launch)]
fn add_sub<F: Float, N: Size>(
    state: Array<Atomic<Vector<F, N>>>,
    values: Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
) {
    out[0] = state[0].fetch_add(values[0]);
    out[1] = state[0].fetch_sub(values[0]);
    out[2] = state[0].load();
}

#[ruda(launch)]
fn min_max<F: Numeric, N: Size>(
    state: Array<Atomic<Vector<F, N>>>,
    values: Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
) {
    out[0] = state[0].fetch_min(values[0]);
    out[1] = state[0].fetch_max(values[1]);
    out[2] = state[0].load();
}

fn require<F: Numeric>(client: &ComputeClient<CudaRuntime>, width: usize, usage: AtomicUsage) {
    let ty = Atomic::<F>::as_type_native_unchecked().with_vector_size(width);
    assert!(
        client.properties().atomic_type_usage(ty).contains(usage),
        "{ty} lacks {usage:?}"
    );
}

fn check_contention<F: Numeric + RudaElement>(width: usize) {
    let client = CudaRuntime::client(&CudaDevice::default());
    require::<F>(&client, width, AtomicUsage::Add);
    for _ in 0..3 {
        let state = client.create_from_slice(F::as_bytes(&vec![F::from_int(0); width]));
        let old = client.empty(256 * width * size_of::<F>());
        unsafe {
            contended_add::launch::<F, CudaRuntime>(
                &client,
                RudaCount::Static(4, 1, 1),
                RudaDim::new_1d(64),
                width,
                ArrayArg::from_raw_parts(state.clone(), width),
                ArrayArg::from_raw_parts(old.clone(), 256 * width),
            );
        }
        let bytes = client.read_one(state).unwrap();
        assert_eq!(F::from_bytes(&bytes), vec![F::from_int(256); width]);
        let bytes = client.read_one(old).unwrap();
        let values = F::from_bytes(&bytes);
        for lane in 0..width {
            let mut tickets: Vec<_> = (0..256).map(|i| values[i * width + lane]).collect();
            tickets.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(tickets, (0..256).map(F::from_int).collect::<Vec<_>>());
        }
    }
}

fn check_shared<F: Float + RudaElement>(width: usize) {
    let client = CudaRuntime::client(&CudaDevice::default());
    require::<F>(&client, width, AtomicUsage::Add);
    require::<F>(&client, width, AtomicUsage::Exchange);
    for native in [false, true] {
        let out = client.empty(69 * width * size_of::<F>());
        unsafe {
            shared_atomic_add::launch::<F, CudaRuntime>(
                &client,
                RudaCount::new_single(),
                RudaDim::new_1d(64),
                width,
                ArrayArg::from_raw_parts(out.clone(), 69 * width),
                native,
            );
        }
        let bytes = client.read_one(out).unwrap();
        let values = F::from_bytes(&bytes);
        for lane in 0..width {
            let mut tickets: Vec<_> = (0..64).map(|i| values[i * width + lane]).collect();
            tickets.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(tickets, (0..64).map(F::from_int).collect::<Vec<_>>());
        }
        let expected: Vec<_> = [64, 64, 63, 63, 71]
            .into_iter()
            .flat_map(|value| vec![F::from_int(value); width])
            .collect();
        assert_eq!(&values[64 * width..], &expected);
    }
}

fn check_bits<F: Numeric + RudaElement>(width: usize, initial: &[F]) {
    let client = CudaRuntime::client(&CudaDevice::default());
    require::<F>(&client, width, AtomicUsage::LoadStore);
    require::<F>(&client, width, AtomicUsage::Exchange);
    let initial: Vec<_> = (0..width).map(|i| initial[i % initial.len()]).collect();
    let guard = vec![F::from_int(73); width];
    let mut state_values = initial.clone();
    state_values.extend(&guard);
    let groups: Vec<Vec<F>> = [1, 5, 5, 20, 40]
        .into_iter()
        .map(|base| (0..width).map(|i| F::from_int(base + i as i64)).collect())
        .collect();
    let mut groups = groups;
    groups[2][width - 1] = F::from_int(99);
    let inputs: Vec<_> = groups.iter().flatten().copied().collect();
    let state = client.create_from_slice(F::as_bytes(&state_values));
    let values = client.create_from_slice(F::as_bytes(&inputs));
    let out = client.empty(10 * width * size_of::<F>());
    unsafe {
        bit_roundtrip::launch::<F, CudaRuntime>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            width,
            ArrayArg::from_raw_parts(state.clone(), 2 * width),
            ArrayArg::from_raw_parts(values, 5 * width),
            ArrayArg::from_raw_parts(out.clone(), 10 * width),
        );
    }
    let expected: Vec<_> = [
        &initial, &initial, &groups[0], &groups[0], &groups[1], &groups[1], &groups[1], &groups[1],
        &groups[4], &guard,
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect();
    assert_eq!(&client.read_one(out).unwrap()[..], F::as_bytes(&expected));
    let expected: Vec<_> = groups[4].iter().chain(guard.iter()).copied().collect();
    assert_eq!(&client.read_one(state).unwrap()[..], F::as_bytes(&expected));
}

fn check_subnormal<F: Float + RudaElement>(width: usize, one: F, two: F) {
    let client = CudaRuntime::client(&CudaDevice::default());
    require::<F>(&client, width, AtomicUsage::Add);
    let state = client.create_from_slice(F::as_bytes(&vec![one; width]));
    let values = client.create_from_slice(F::as_bytes(&vec![one; width]));
    let out = client.empty(3 * width * size_of::<F>());
    unsafe {
        add_sub::launch::<F, CudaRuntime>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            width,
            ArrayArg::from_raw_parts(state, width),
            ArrayArg::from_raw_parts(values, width),
            ArrayArg::from_raw_parts(out.clone(), 3 * width),
        );
    }
    let expected: Vec<_> = [one, two, one]
        .into_iter()
        .flat_map(|v| vec![v; width])
        .collect();
    assert_eq!(&client.read_one(out).unwrap()[..], F::as_bytes(&expected));
}

fn check_minmax<F: Numeric + RudaElement>(width: usize) {
    let client = CudaRuntime::client(&CudaDevice::default());
    require::<F>(&client, width, AtomicUsage::MinMax);
    let unsigned = F::min_value() == F::from_int(0);
    let offset = if unsigned { width as i64 + 2 } else { 0 };
    let mut initial: Vec<_> = (0..width)
        .map(|i| F::from_int(offset + i as i64 - 2))
        .collect();
    if unsigned {
        initial[0] = F::max_value();
    }
    let lower: Vec<_> = (0..width).map(|i| F::from_int(offset - i as i64)).collect();
    let upper: Vec<_> = (0..width)
        .map(|i| F::from_int(offset + i as i64 + 1))
        .collect();
    let middle: Vec<_> = initial
        .iter()
        .zip(&lower)
        .map(|(a, b)| if a < b { *a } else { *b })
        .collect();
    let state = client.create_from_slice(F::as_bytes(&initial));
    let inputs: Vec<_> = lower.into_iter().chain(upper.iter().copied()).collect();
    let values = client.create_from_slice(F::as_bytes(&inputs));
    let out = client.empty(3 * width * size_of::<F>());
    unsafe {
        min_max::launch::<F, CudaRuntime>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            width,
            ArrayArg::from_raw_parts(state, width),
            ArrayArg::from_raw_parts(values, 2 * width),
            ArrayArg::from_raw_parts(out.clone(), 3 * width),
        );
    }
    let final_values: Vec<_> = middle
        .iter()
        .zip(&upper)
        .map(|(a, b)| if a > b { *a } else { *b })
        .collect();
    let expected: Vec<_> = initial
        .into_iter()
        .chain(middle)
        .chain(final_values)
        .collect();
    assert_eq!(&client.read_one(out).unwrap()[..], F::as_bytes(&expected));
}

macro_rules! half_cases {
    ($name:ident, $ty:ty) => {
        mod $name {
            use super::*;
            #[test]
            fn shared_and_native_pointer_atomics() {
                for width in [1, 2, 4, 8] {
                    check_shared::<$ty>(width);
                }
            }
            #[test]
            fn contention() {
                for width in [1, 2, 4, 8] {
                    check_contention::<$ty>(width);
                }
            }
            #[test]
            fn bitwise_load_store_exchange_cas() {
                for width in [1, 2, 4, 8] {
                    check_bits::<$ty>(
                        width,
                        &[
                            <$ty>::from_bits(0x8000),
                            <$ty>::from_bits(0x7fc1),
                            <$ty>::from_bits(1),
                        ],
                    );
                }
            }
            #[test]
            fn subnormal_add_sub() {
                for width in [1, 2, 4, 8] {
                    check_subnormal::<$ty>(width, <$ty>::from_bits(1), <$ty>::from_bits(2));
                }
            }
            #[test]
            fn vector_minmax() {
                for width in [4, 8] {
                    check_minmax::<$ty>(width);
                }
            }
        }
    };
}
half_cases!(fp16, f16);
half_cases!(bfloat16, bf16);

#[test]
fn f32_shared_and_native_pointer_atomics() {
    for width in [1, 2, 4] {
        check_shared::<f32>(width);
    }
}
#[test]
fn f64_shared_and_native_pointer_atomics() {
    check_shared::<f64>(1);
}

#[test]
fn f32_contention() {
    for width in [1, 2, 4] {
        check_contention::<f32>(width);
    }
}
#[test]
fn f32_bitwise_load_store_exchange_cas() {
    for width in [1, 2, 4] {
        check_bits::<f32>(
            width,
            &[-0.0, f32::from_bits(0x7fc00042), f32::from_bits(1)],
        );
    }
}
#[test]
fn f64_contention() {
    check_contention::<f64>(1);
}
#[test]
fn f64_bitwise_load_store_exchange_cas() {
    check_bits::<f64>(1, &[-0.0]);
}

#[test]
fn signed_and_unsigned_integer_atomics() {
    macro_rules! check {
        ($ty:ty) => {
            check_contention::<$ty>(1);
            check_bits::<$ty>(1, &[<$ty>::MAX]);
            check_minmax::<$ty>(1);
        };
    }
    check!(i32);
    check!(i64);
    check!(u32);
    check!(u64);
}
