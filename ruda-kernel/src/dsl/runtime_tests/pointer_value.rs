use crate::dsl::prelude::*;

#[ruda]
fn pointer_value_helper(pointer: ValuePtr<f32, GlobalSpace>) -> ValuePtr<f32, GlobalSpace> {
    pointer.offset(0)
}

#[ruda(launch)]
fn pointer_value_kernel(input: &Array<f32>, other: &Array<f32>, output: &mut Array<f32>) {
    let index = UNIT_POS as usize;
    let first = input.to_slice().typed::<GlobalSpace>().ptr(0).value();
    let second = other.to_slice().typed::<GlobalSpace>().ptr(0).value();
    let mut pointer = if index % 2 == 0 { first } else { second };
    let mut sum = 0.0f32;
    for _i in 0..index + 1 {
        sum += pointer_value_helper(pointer).load();
        pointer = pointer.offset(1);
    }
    let selected = second.select(index < 2, first).offset(index);
    let mut local = Array::<f32>::new(2usize);
    let local_view = local.to_slice_mut().typed::<Local>();
    local_view.ptr(0).value().store(sum);
    local_view.ptr(1).value().store(selected.load());
    let mut local_pointer = local_view.ptr(0).value();
    if index % 2 == 0 {
        local_pointer = local_view.ptr(1).value();
    }
    let mut shared = SharedMemory::<f32>::new(4usize);
    let shared_pointer = shared
        .to_slice_mut()
        .typed::<Workgroup>()
        .ptr(index)
        .value();
    shared_pointer.store(local_pointer.read_only().load());
    sync_ruda();
    output
        .to_slice_mut()
        .typed::<GlobalSpace>()
        .ptr(index)
        .value()
        .store(sum + shared_pointer.load());
}

pub fn test_branch_loop_select_and_spaces<R: Runtime>() {
    let client = R::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let other = client.create_from_slice(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let output = client.empty(4 * core::mem::size_of::<f32>());
    unsafe {
        pointer_value_kernel::launch::<R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(input, 4),
            ArrayArg::from_raw_parts(other, 4),
            ArrayArg::from_raw_parts(output.clone(), 4),
        );
    }
    assert_eq!(
        f32::from_bytes(&client.read_one_unchecked(output)),
        &[11.0, 60.0, 9.0, 200.0]
    );
}

#[ruda(launch)]
fn pointer_value_launch_kernel(
    input: ValuePtr<f32, GlobalSpace>,
    output: &mut ValuePtr<f32, GlobalSpace, ReadWrite>,
) {
    let index = UNIT_POS as usize;
    output
        .offset(index)
        .store(pointer_value_helper(input).offset(index).load() + 1.0);
}

pub fn test_kernel_arguments<R: Runtime>(inplace: bool) {
    let client = R::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = if inplace {
        input.clone()
    } else {
        client.empty(4 * core::mem::size_of::<f32>())
    };
    unsafe {
        let argument = if inplace {
            ArrayArg::Alias {
                input_pos: 0,
                length: [4],
            }
        } else {
            ArrayArg::from_raw_parts(output.clone(), 4)
        };
        pointer_value_launch_kernel::launch::<R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(input, 4),
            argument,
        );
    }
    assert_eq!(
        f32::from_bytes(&client.read_one_unchecked(output)),
        &[2.0, 3.0, 4.0, 5.0]
    );
}

#[ruda(launch)]
fn pointer_value_atomic_kernel(data: &mut Array<Atomic<u32>>, out: &mut Array<u32>) {
    if UNIT_POS == 0 {
        let pointer = data.to_slice_mut().typed::<GlobalSpace>().ptr(0).value();
        pointer.atomic_store(1);
        out[0] = pointer.read_only().atomic_load();
        out[1] = pointer.swap(2);
        out[2] = pointer.fetch_add(3);
        out[3] = pointer.fetch_sub(1);
        out[4] = pointer.fetch_min(2);
        out[5] = pointer.fetch_max(4);
        out[6] = pointer.fetch_and(3);
        out[7] = pointer.fetch_or(4);
        out[8] = pointer.fetch_xor(1);
        out[9] = pointer.compare_exchange_weak(5, 7);
        while pointer.read_only().atomic_load() == 5 {
            let _retry = pointer.compare_exchange_weak(5, 7);
        }
        out[10] = pointer.read_only().atomic_load();
    }
}

pub fn test_atomics<R: Runtime>() {
    use crate::dsl::ir::{ElemType, StorageType, Type, UIntKind, features::AtomicUsage};
    let client = R::client(&Default::default());
    let ty = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
    let supported = client.properties().atomic_type_usage(ty);
    if !supported.contains(AtomicUsage::LoadStore) || !supported.contains(AtomicUsage::Add) {
        println!("pointer_value atomics not supported by this runtime; not executed");
        return;
    }
    let data = client.empty(core::mem::size_of::<u32>());
    let output = client.empty(11 * core::mem::size_of::<u32>());
    unsafe {
        pointer_value_atomic_kernel::launch::<R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(data, 1),
            ArrayArg::from_raw_parts(output.clone(), 11),
        );
    }
    assert_eq!(
        u32::from_bytes(&client.read_one_unchecked(output)),
        &[1, 1, 2, 5, 4, 2, 4, 0, 4, 5, 7]
    );
}

#[ruda(launch)]
fn pointer_value_native_kernel(output: &mut Array<f32>) {
    let index = UNIT_POS as usize;
    let mut local = Array::<f32>::new(1usize);
    let local_pointer = local.to_slice_mut().typed::<Local>().ptr(0).value();
    let local_native =
        ValuePtr::<f32, Local, ReadWrite>::from_native(local_pointer.native_address());
    local_native.store((index as f32) + 1.0);
    let mut shared = SharedMemory::<f32>::new(4usize);
    let shared_pointer = shared.to_slice_mut().typed::<Workgroup>().ptr(0).value();
    let shared_native =
        ValuePtr::<f32, Workgroup, ReadWrite>::from_native(shared_pointer.native_address())
            .offset(index);
    shared_native.store(local_native.read_only().load() * 2.0);
    sync_ruda();
    let pointer = output.to_slice_mut().typed::<GlobalSpace>().ptr(0).value();
    let imported = ValuePtr::<f32, GlobalSpace, ReadWrite>::from_native(pointer.native_address())
        .offset(index);
    imported.store(shared_native.read_only().load());
    output[index] = imported.read_only().load() * 3.0;
}

pub fn test_native<R: Runtime>() {
    let client = R::client(&Default::default());
    let output = client.empty(4 * core::mem::size_of::<f32>());
    unsafe {
        pointer_value_native_kernel::launch::<R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(output.clone(), 4),
        );
    }
    assert_eq!(
        f32::from_bytes(&client.read_one_unchecked(output)),
        &[6.0, 12.0, 18.0, 24.0]
    );
}

#[macro_export]
macro_rules! testgen_pointer_value {
    () => {
        #[test]
        fn test_pointer_value_branch_loop_select_and_spaces() {
            $crate::dsl::runtime_tests::pointer_value::test_branch_loop_select_and_spaces::<
                TestRuntime,
            >();
        }
        #[test]
        fn test_pointer_value_kernel_arguments() {
            $crate::dsl::runtime_tests::pointer_value::test_kernel_arguments::<TestRuntime>(false);
        }
        #[test]
        fn test_pointer_value_inplace_kernel_arguments() {
            $crate::dsl::runtime_tests::pointer_value::test_kernel_arguments::<TestRuntime>(true);
        }
        #[test]
        fn test_pointer_value_atomics() {
            $crate::dsl::runtime_tests::pointer_value::test_atomics::<TestRuntime>();
        }
    };
}
