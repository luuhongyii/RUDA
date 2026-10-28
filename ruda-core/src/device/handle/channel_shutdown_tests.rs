use super::*;
use std::string::String;
use std::sync::{
    Arc, Barrier, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

fn client(index: u16) -> DeviceClient {
    DeviceClient::new(
        RunnerId {
            device: DeviceId::new(64999, index),
            stage: DeviceServiceStage::Downstream,
        },
        || {},
    )
}

#[test]
fn partial_queue_is_drained_and_closed_handles_reject_new_work() {
    let client = client(1);
    let count = Arc::new(AtomicUsize::new(0));
    for _ in 0..17 {
        let count = count.clone();
        client
            .enqueue(move || {
                count.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
    }
    client.shutdown();
    assert_eq!(count.load(Ordering::Relaxed), 17);
    assert!(
        client
            .enqueue(|| panic!("closed work must not execute"))
            .is_err()
    );
    client.flush();
    client.shutdown();
}

#[test]
fn concurrent_close_preserves_every_accepted_submission() {
    let client = client(2);
    let accepted = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(5));
    let mut workers = Vec::new();
    for _ in 0..4 {
        let client = client.clone();
        let (accepted, executed, barrier) = (accepted.clone(), executed.clone(), barrier.clone());
        workers.push(std::thread::spawn(move || {
            let first = executed.clone();
            client
                .enqueue(move || {
                    first.fetch_add(1, Ordering::Relaxed);
                })
                .unwrap();
            accepted.fetch_add(1, Ordering::Relaxed);
            barrier.wait();
            for _ in 0..1000 {
                let executed = executed.clone();
                if client
                    .enqueue(move || {
                        executed.fetch_add(1, Ordering::Relaxed);
                    })
                    .is_err()
                {
                    break;
                }
                accepted.fetch_add(1, Ordering::Relaxed);
                client.flush();
            }
        }));
    }
    barrier.wait();
    client.shutdown();
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(accepted.load(Ordering::Relaxed) >= 4);
    assert_eq!(
        accepted.load(Ordering::Relaxed),
        executed.load(Ordering::Relaxed)
    );
}

struct Finisher {
    work: usize,
    owner: std::thread::ThreadId,
    finished: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}
impl DeviceService for Finisher {
    fn init(_: DeviceId) -> Self {
        unreachable!()
    }
    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }
    fn shutdown(&mut self) {
        assert_eq!(self.owner, std::thread::current().id());
        assert_eq!(self.work, 7);
        self.finished.store(true, Ordering::Release);
    }
}
impl Drop for Finisher {
    fn drop(&mut self) {
        assert_eq!(self.owner, std::thread::current().id());
        assert!(self.finished.load(Ordering::Acquire));
        self.dropped.store(true, Ordering::Release);
    }
}
#[test]
fn service_finishes_and_drops_before_join_returns() {
    let (finished, dropped) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let handle = ChannelDeviceHandle::insert(
        DeviceId::new(64998, 1),
        Finisher {
            work: 0,
            owner: std::thread::current().id(),
            finished: finished.clone(),
            dropped: dropped.clone(),
        },
    )
    .unwrap();
    handle
        .submit_blocking(|s| s.owner = std::thread::current().id())
        .unwrap();
    for _ in 0..7 {
        handle.submit(|s| s.work += 1);
    }
    handle.state.client.shutdown();
    assert!(finished.load(Ordering::Acquire));
    assert!(dropped.load(Ordering::Acquire));
}

static EVENTS: Mutex<Vec<&str>> = Mutex::new(Vec::new());
fn event(value: &'static str) {
    EVENTS.lock().unwrap().push(value);
}
struct Downstream;
impl DeviceService for Downstream {
    fn init(_: DeviceId) -> Self {
        Self
    }
    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }
    fn shutdown(&mut self) {
        event("down_shutdown");
    }
}
impl Drop for Downstream {
    fn drop(&mut self) {
        event("down_drop");
    }
}
struct Upstream(ChannelDeviceHandle<Downstream>);
impl DeviceService for Upstream {
    fn init(_: DeviceId) -> Self {
        unreachable!()
    }
    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }
    fn stage() -> DeviceServiceStage {
        DeviceServiceStage::Upstream
    }
    fn shutdown(&mut self) {
        event("up_shutdown");
        self.0.submit(|_| event("down_from_shutdown"));
    }
}
impl Drop for Upstream {
    fn drop(&mut self) {
        event("up_drop");
        self.0.submit(|_| event("down_from_drop"));
    }
}

#[cfg(any(unix, windows))]
#[test]
fn process_exit_child() {
    let Ok(mode) = std::env::var("RUDA_SHUTDOWN_CHILD") else {
        return;
    };
    extern "C" fn observe() {
        let path = std::env::var("RUDA_SHUTDOWN_OUTPUT").unwrap();
        std::fs::write(path, EVENTS.lock().unwrap().join(",")).unwrap();
    }
    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> core::ffi::c_int;
        fn exit(code: core::ffi::c_int) -> !;
    }
    // SAFETY: static C callback; this isolated child reports the exit ordering to its parent.
    assert_eq!(unsafe { atexit(observe) }, 0);
    let down = ChannelDeviceHandle::<Downstream>::new(DeviceId::new(64997, 1));
    let up = ChannelDeviceHandle::insert(DeviceId::new(64997, 1), Upstream(down)).unwrap();
    up.submit(|_| event("up_work"));
    if mode == "exit" {
        std::process::exit(0);
    }
    if mode == "c-exit" {
        // Exercise the CRT hooks directly; Rust uses ExitProcess on Windows.
        unsafe { exit(0) };
    }
}

#[cfg(any(unix, windows))]
fn check_child(mode: &str) {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("events.txt");
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "device::handle::channel::shutdown_tests::process_exit_child",
            "--exact",
            "--nocapture",
        ])
        .env("RUDA_SHUTDOWN_CHILD", mode)
        .env("RUDA_SHUTDOWN_OUTPUT", &output)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "{}",
        String::from_utf8_lossy(&child.stderr)
    );
    if cfg!(windows) && mode == "exit" {
        assert!(!output.exists(), "ExitProcess does not invoke CRT atexit handlers");
        return;
    }
    assert_eq!(
        std::fs::read_to_string(output).unwrap(),
        "up_work,up_shutdown,up_drop,down_from_shutdown,down_from_drop,down_shutdown,down_drop"
    );
    assert!(!String::from_utf8_lossy(&child.stderr).contains("panicked"));
}
#[cfg(any(unix, windows))]
#[test]
fn normal_return_joins_pipeline_before_other_exit_handlers() {
    check_child("return");
}
#[cfg(any(unix, windows))]
#[test]
fn explicit_process_exit_matches_platform_exit_hooks() {
    check_child("exit");
}

#[cfg(any(unix, windows))]
#[test]
fn explicit_c_exit_joins_pipeline_before_other_exit_handlers() {
    check_child("c-exit");
}
