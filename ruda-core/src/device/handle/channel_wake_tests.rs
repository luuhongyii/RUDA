use super::*;
use std::sync::{Arc, Barrier, atomic::{AtomicUsize, Ordering}};
use std::time::Duration;

fn client(index: u16) -> DeviceClient {
    DeviceClient::new(RunnerId {
        device: DeviceId::new(64997, index), stage: DeviceServiceStage::Downstream,
    }, || {})
}

#[test]
fn idle_runner_wakes_for_repeated_partial_flushes() {
    let client = client(1);
    for value in 0..8 {
        std::thread::sleep(Duration::from_millis(2));
        let (send, receive) = oneshot::channel();
        client.enqueue(move || send.send(value).unwrap()).unwrap();
        client.flush();
        assert_eq!(receive.recv_timeout(Duration::from_secs(2)).unwrap(), value);
    }
    client.shutdown();
}

#[test]
fn idle_runner_wakes_for_an_automatically_completed_batch() {
    let client = client(2);
    let count = Arc::new(AtomicUsize::new(0));
    std::thread::sleep(Duration::from_millis(2));
    for index in 0..custom_channel::CHANNEL_MAX_TASK - 1 {
        let count = count.clone();
        client.enqueue(move || { assert_eq!(count.fetch_add(1, Ordering::SeqCst), index); }).unwrap();
    }
    let (send, receive) = oneshot::channel();
    let completed = count.clone();
    client.enqueue(move || send.send(completed.fetch_add(1, Ordering::SeqCst)).unwrap()).unwrap();
    assert_eq!(receive.recv_timeout(Duration::from_secs(2)).unwrap(), custom_channel::CHANNEL_MAX_TASK - 1);
    assert_eq!(count.load(Ordering::SeqCst), custom_channel::CHANNEL_MAX_TASK);
    client.shutdown();
}

#[test]
fn concurrent_wakes_and_flushes_keep_every_producer_ordered() {
    let client = client(3);
    let barrier = Arc::new(Barrier::new(5));
    let counts = Arc::new([const { AtomicUsize::new(0) }; 4]);
    let mut workers = Vec::new();
    for producer in 0..4 {
        let (client, barrier, counts) = (client.clone(), barrier.clone(), counts.clone());
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            for index in 0..200 {
                let counts = counts.clone();
                client.enqueue(move || {
                    assert_eq!(counts[producer].fetch_add(1, Ordering::SeqCst), index);
                }).unwrap();
                if index % 7 == 0 { client.flush(); }
            }
        }));
    }
    barrier.wait();
    for worker in workers { worker.join().unwrap(); }
    let (send, receive) = oneshot::channel();
    client.enqueue(move || send.send(()).unwrap()).unwrap();
    client.flush();
    receive.recv_timeout(Duration::from_secs(2)).unwrap();
    for count in counts.iter() { assert_eq!(count.load(Ordering::SeqCst), 200); }
    client.shutdown();
}

#[test]
fn shutdown_releases_an_idle_parked_runner() {
    let client = client(4);
    std::thread::sleep(Duration::from_millis(2));
    let (send, receive) = oneshot::channel();
    let closer = std::thread::spawn(move || { client.shutdown(); send.send(()).unwrap(); });
    receive.recv_timeout(Duration::from_secs(2)).unwrap();
    closer.join().unwrap();
}
