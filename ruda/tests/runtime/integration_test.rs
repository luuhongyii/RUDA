#![allow(unsafe_code)]

mod dummy;

use crate::dummy::{DummyDevice, DummyElementwiseAddition, test_client};

use ruda::runtime::server::RudaCount;
use ruda::runtime::server::KernelArguments;
use ruda::runtime::{local_tuner, tune::LocalTuner};
use dummy::*;

#[test_log::test]
fn created_resource_is_the_same_when_read() {
    let client = test_client(&DummyDevice);
    let resource = Vec::from([0, 1, 2]);
    let resource_description = client.create_from_slice(&resource);

    let obtained_resource = client.read_one(resource_description).unwrap().to_vec();

    assert_eq!(resource, obtained_resource)
}

#[test_log::test]
fn empty_allocates_memory() {
    let client = test_client(&DummyDevice);
    let size = 4;
    let resource_description = client.empty(size);
    let empty_resource = client.read_one(resource_description).unwrap();

    assert_eq!(empty_resource.len(), 4);
}

#[test_log::test]
fn empty_with_resource_preserves_layout_and_handle_lifetime() {
    let client = test_client(&DummyDevice);
    for size in [0, 1, 3, 4, 31, 32, 33, 4097] {
        let (handle, resource) = client.try_empty_with_resource(size).unwrap();
        assert_eq!(handle.size_in_used(), size as u64);
        assert_eq!(handle.service, client.service_id());
        assert_eq!(handle.stream, client.execution_stream());
        resource.resource().write()[..size].fill(73);
        let lookup = client.get_resource(handle.clone()).unwrap();
        assert_eq!(&lookup.resource().read()[..size], &vec![73; size]);
        drop(resource);
        drop(lookup);
        assert_eq!(client.read_one(handle).unwrap().to_vec(), vec![73; size]);
        let handle = client.try_empty(size).unwrap();
        assert_eq!(client.read_one(handle).unwrap().len(), size);
    }
}

#[test_log::test]
fn empty_with_resource_retains_storage_after_handle_drop() {
    let client = test_client(&DummyDevice);
    let (handle, resource) = client.try_empty_with_resource(4097).unwrap();
    resource.resource().write()[..4097].fill(29);
    drop(handle);
    for _ in 0..64 {
        let (other, scratch) = client.try_empty_with_resource(4097).unwrap();
        scratch.resource().write()[..4097].fill(91);
        drop(other);
        assert_eq!(&resource.resource().read()[..4097], &vec![29; 4097]);
    }
}

#[test_log::test]
fn empty_with_resource_uses_selected_stream() {
    let stream = ruda_core::stream_id::StreamId::unique();
    stream.executes(|| {
        let client = test_client(&DummyDevice).fixed_execution_queue();
        let (handle, resource) = client.try_empty_with_resource(33).unwrap();
        assert_eq!(handle.stream, stream);
        resource.resource().write()[..33].fill(17);
        drop(resource);
        assert_eq!(client.read_one(handle).unwrap().to_vec(), vec![17; 33]);
    });
}

#[test_log::test]
fn execute_elementwise_addition() {
    let client = test_client(&DummyDevice);
    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);

    client.launch(
        Box::new(KernelTask::new(DummyElementwiseAddition)),
        RudaCount::Static(1, 1, 1),
        KernelArguments::new().with_buffers(vec![
            lhs.binding(),
            rhs.binding(),
            out.clone().binding(),
        ]),
    );

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    assert_eq!(obtained_resource, Vec::from([4, 5, 6]))
}

#[test_log::test]
#[cfg(feature = "runtime-std")]
fn autotune_basic_addition_execution() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_basic_addition_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([4, 5, 6]));
}

#[test_log::test]
#[cfg(feature = "runtime-std")]
fn autotune_basic_multiplication_execution() {
    static TUNER: LocalTuner<String, String> =
        local_tuner!("autotune_basic_multiplication_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::multiplication_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([0, 4, 8]));
}
