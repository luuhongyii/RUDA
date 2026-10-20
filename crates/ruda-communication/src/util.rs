use tracing_core::{Level, LevelFilter};
use tracing_subscriber::{
    Layer, filter::filter_fn, layer::SubscriberExt, registry, util::SubscriberInitExt,
};

/// Utilities to help handle communication termination.
pub async fn os_shutdown_signal() {
    try_os_shutdown_signal().await.expect("failed to install shutdown signal handler");
}

/// Wait for process shutdown, reporting signal-handler installation failures.
pub async fn try_os_shutdown_signal() -> std::io::Result<()> {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?
            .recv().await;
        Ok(())
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<std::io::Result<()>>();

    tokio::select! {
        result = ctrl_c => result,
        result = terminate => result,
    }
}

pub(crate) fn init_logging() {
    let layer = tracing_subscriber::fmt::layer()
        .with_filter(LevelFilter::INFO)
        .with_filter(filter_fn(|m| {
            if let Some(path) = m.module_path() {
                // The wgpu crate is logging too much, so we skip `info` level.
                if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                    return false;
                }
            }
            true
        }));

    // If we start multiple servers in the same process, this will fail, it's ok
    let _ = registry().with(layer).try_init();
}
