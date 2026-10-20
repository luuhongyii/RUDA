use super::{Metric, NumericEntry, SerializedEntry};
use crate::TrainingError;
use nvml_wrapper::Nvml;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use sysinfo::{CpuRefreshKind, RefreshKind, System};
use systemstat::Platform;

macro_rules! instantaneous_metric {
    () => {
        fn current_value(&self) -> Option<NumericEntry> {
            self.value().map(NumericEntry::Value)
        }
        fn running_value(&self) -> Option<NumericEntry> {
            self.current_value()
        }
        fn final_value(&self) -> NumericEntry {
            NumericEntry::Value(self.value().unwrap_or(f64::NAN))
        }
        fn serialized_final(&self) -> SerializedEntry {
            self.serialized_update()
        }
        fn clear(&mut self) {}
    };
}

pub struct CpuUse {
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    current: f64,
}
impl CpuUse {
    pub fn new() -> Self {
        let mut sys = System::new();
        let current = Self::refresh(&mut sys);
        Self {
            last_refresh: Instant::now(),
            refresh_frequency: Duration::from_millis(200),
            sys,
            current,
        }
    }
    fn refresh(sys: &mut System) -> f64 {
        sys.refresh_specifics(
            RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing().with_cpu_usage()),
        );
        let cpus = sys.cpus();
        let usage = cpus.iter().fold(0., |acc, cpu| acc + cpu.cpu_usage()) as f64;
        usage / cpus.len() as f64
    }
}
impl Default for CpuUse {
    fn default() -> Self {
        Self::new()
    }
}
impl Clone for CpuUse {
    fn clone(&self) -> Self {
        Self {
            last_refresh: self.last_refresh,
            refresh_frequency: self.refresh_frequency,
            sys: System::new(),
            current: self.current,
        }
    }
}
impl Metric for CpuUse {
    numeric_attributes!(Some("%"), false, 2);
    instantaneous_metric!();
    type Input = ();
    fn name(&self) -> &str {
        "CPU Usage"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        if self.last_refresh.elapsed() >= self.refresh_frequency {
            self.current = Self::refresh(&mut self.sys);
            self.last_refresh = Instant::now();
        }
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        Some(self.current)
    }
    fn serialized_update(&self) -> SerializedEntry {
        SerializedEntry::new(
            format!("{}: {:.2} %", self.name(), self.current),
            format!("{:.2}", self.current),
        )
    }
}

#[derive(Clone)]
pub struct CpuTemperature {
    temp_celsius: f32,
    sys: Arc<systemstat::System>,
}
impl CpuTemperature {
    pub fn new() -> Self {
        Self {
            temp_celsius: 0.,
            sys: Arc::new(systemstat::System::new()),
        }
    }
}
impl Default for CpuTemperature {
    fn default() -> Self {
        Self::new()
    }
}
impl Metric for CpuTemperature {
    numeric_attributes!(Some("°C"), false, 2);
    instantaneous_metric!();
    type Input = ();
    fn name(&self) -> &str {
        "CPU Temperature"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        self.compute()
    }
    fn compute(&mut self) -> Result<(), TrainingError> {
        self.temp_celsius = self.sys.cpu_temp().unwrap_or(f32::NAN);
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        Some(self.temp_celsius as f64)
    }
    fn serialized_update(&self) -> SerializedEntry {
        SerializedEntry::new(
            format!("{}: {:.2} °C", self.name(), self.temp_celsius),
            format!("{:.2}", self.temp_celsius),
        )
    }
}

pub struct CpuMemory {
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    ram_bytes_total: u64,
    ram_bytes_used: u64,
}
impl CpuMemory {
    pub fn new() -> Self {
        let mut metric = Self {
            last_refresh: Instant::now(),
            refresh_frequency: Duration::from_millis(200),
            sys: System::new(),
            ram_bytes_total: 0,
            ram_bytes_used: 0,
        };
        metric.refresh();
        metric
    }
    fn refresh(&mut self) {
        self.sys.refresh_memory();
        self.last_refresh = Instant::now();
        self.ram_bytes_total = self.sys.total_memory();
        self.ram_bytes_used = self.sys.used_memory();
    }
}
impl Default for CpuMemory {
    fn default() -> Self {
        Self::new()
    }
}
impl Clone for CpuMemory {
    fn clone(&self) -> Self {
        Self {
            last_refresh: self.last_refresh,
            refresh_frequency: self.refresh_frequency,
            sys: System::new(),
            ram_bytes_total: self.ram_bytes_total,
            ram_bytes_used: self.ram_bytes_used,
        }
    }
}
impl Metric for CpuMemory {
    numeric_attributes!(Some("Gb"), false, 2);
    instantaneous_metric!();
    type Input = ();
    fn name(&self) -> &str {
        "CPU Memory"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        if self.last_refresh.elapsed() >= self.refresh_frequency {
            self.refresh();
        }
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        Some(self.ram_bytes_used as f64 / 1e9)
    }
    fn serialized_update(&self) -> SerializedEntry {
        let used = self.ram_bytes_used as f64 / 1e9;
        SerializedEntry::new(
            format!(
                "RAM Used: {:.2} / {:.2} Gb",
                used,
                self.ram_bytes_total as f64 / 1e9
            ),
            used.to_string(),
        )
    }
}

#[derive(Clone)]
pub struct CudaMetric {
    nvml: Arc<Option<Nvml>>,
    entry: SerializedEntry,
}
impl CudaMetric {
    pub fn new() -> Self {
        Self {
            nvml: Arc::new(Nvml::init().map(Some).unwrap_or_else(|error| {
                log::warn!("Unable to initialize CUDA Metric: {error}");
                None
            })),
            entry: Self::unavailable(),
        }
    }
    fn unavailable() -> SerializedEntry {
        SerializedEntry::new("Unavailable".into(), "Unavailable".into())
    }
    fn read(nvml: &Nvml) -> SerializedEntry {
        let mut formatted = String::new();
        let mut raw = String::new();
        let device_count = match nvml.device_count() {
            Ok(count) => count,
            Err(error) => {
                log::warn!("Unable to get the number of cuda devices: {error}");
                return Self::unavailable();
            }
        };
        for index in 0..device_count {
            let device = match nvml.device_by_index(index) {
                Ok(device) => device,
                Err(error) => {
                    log::warn!("Unable to get device {index}: {error}");
                    return Self::unavailable();
                }
            };
            let memory = match device.memory_info() {
                Ok(memory) => memory,
                Err(error) => {
                    log::warn!("Unable to get memory info from device {index}: {error}");
                    return Self::unavailable();
                }
            };
            let used = memory.used as f64 * 1e-9;
            let total = memory.total as f64 * 1e-9;
            formatted = format!("{formatted} GPU #{index} - Memory {used:.2}/{total:.2} Gb");
            raw = format!("{used}/{total} ");
            let usage = match device.utilization_rates() {
                Ok(usage) => usage,
                Err(error) => {
                    log::warn!("Unable to get utilization rates from device {index}: {error}");
                    return Self::unavailable();
                }
            };
            formatted = format!("{formatted} - Usage {}%", usage.gpu);
            if let Ok(power) = device.power_usage() {
                formatted = format!("{formatted} - Power {:.1} W", power as f64 / 1000.);
            }
        }
        SerializedEntry::new(formatted, raw)
    }
}
impl Default for CudaMetric {
    fn default() -> Self {
        Self::new()
    }
}
impl Metric for CudaMetric {
    fn is_numeric(&self) -> bool { false }
    type Input = ();
    fn name(&self) -> &str {
        "Cuda"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        self.compute()
    }
    fn compute(&mut self) -> Result<(), TrainingError> {
        self.entry = match self.nvml.as_ref() {
            Some(nvml) => Self::read(nvml),
            None => Self::unavailable(),
        };
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        None
    }
    fn serialized_update(&self) -> SerializedEntry {
        self.entry.clone()
    }
    fn serialized_final(&self) -> SerializedEntry {
        self.entry.clone()
    }
    fn snapshot(&self, _: bool) -> super::MetricSnapshot {
        super::MetricSnapshot {
            definition: self.definition(),
            current: None,
            running: None,
            final_value: None,
            serialized: self.entry.clone(),
        }
    }
    fn clear(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_usage_reads_real_sample_and_keeps_instantaneous_value() {
        let mut metric = CpuUse::new();
        metric.last_refresh = Instant::now() - Duration::from_secs(1);
        metric.update(&()).unwrap();
        assert!(!metric.sys.cpus().is_empty());
        let value = metric.value().unwrap();
        assert!(value.is_finite() && (0. ..=100.).contains(&value));
        assert_eq!(metric.current_value(), Some(NumericEntry::Value(value)));
        assert_eq!(metric.running_value(), metric.current_value());
        assert_eq!(metric.final_value(), NumericEntry::Value(value));
        assert_eq!(metric.clone().value(), Some(value));
        let entry = metric.serialized_update();
        assert_eq!(entry.formatted, format!("CPU Usage: {value:.2} %"));
        metric.clear();
        assert_eq!(metric.value(), Some(value));
        println!(
            "SYSTEM_CPU_USAGE percent={value} cpus={}",
            metric.sys.cpus().len()
        );
    }

    #[test]
    fn memory_reads_real_bytes_and_uses_decimal_gigabytes() {
        let mut metric = CpuMemory::new();
        metric.last_refresh = Instant::now() - Duration::from_secs(1);
        metric.update(&()).unwrap();
        assert!(metric.ram_bytes_total > 0);
        assert!(metric.ram_bytes_used <= metric.ram_bytes_total);
        let value = metric.ram_bytes_used as f64 / 1e9;
        assert_eq!(metric.value(), Some(value));
        assert_eq!(metric.final_value(), NumericEntry::Value(value));
        assert_eq!(metric.serialized_update().serialized, value.to_string());
        assert_eq!(metric.clone().value(), Some(value));
        metric.clear();
        assert_eq!(metric.value(), Some(value));
        println!(
            "SYSTEM_MEMORY used_bytes={} total_bytes={}",
            metric.ram_bytes_used, metric.ram_bytes_total
        );
    }

    #[test]
    fn temperature_reports_real_sensor_or_nan_without_blocking() {
        let mut metric = CpuTemperature::new();
        metric.update(&()).unwrap();
        let value = metric.value().unwrap();
        assert!(value.is_finite() || value.is_nan());
        let entry = metric.serialized_update();
        assert_eq!(entry.serialized, format!("{:.2}", metric.temp_celsius));
        assert_eq!(metric.clone().serialized_update(), entry);
        metric.clear();
        assert_eq!(metric.serialized_update(), entry);
        println!(
            "SYSTEM_TEMPERATURE available={} celsius={value}",
            value.is_finite()
        );
    }

    #[test]
    fn cuda_nvml_reports_actual_memory_usage_and_optional_power() {
        let mut metric = CudaMetric::new();
        metric.update(&()).unwrap();
        let entry = metric.serialized_update();
        if std::env::var("RUDA_REQUIRE_GPU").as_deref() == Ok("1") {
            let nvml = metric
                .nvml
                .as_ref()
                .as_ref()
                .expect("GPU acceptance requires NVML");
            let count = nvml.device_count().unwrap();
            assert!(count > 0);
            assert_ne!(entry.serialized, "Unavailable");
            for index in 0..count {
                let device = nvml.device_by_index(index).unwrap();
                let memory = device.memory_info().unwrap();
                assert!(memory.total > 0 && memory.used <= memory.total);
                assert!(entry.formatted.contains(&format!("GPU #{index} - Memory")));
                assert!(
                    entry
                        .formatted
                        .contains(&format!("/{:.2} Gb", memory.total as f64 * 1e-9))
                );
                if device.power_usage().is_ok() {
                    assert!(entry.formatted.contains(" - Power "));
                }
            }
            assert!(entry.formatted.contains(" - Usage "));
        } else if metric.nvml.is_none() {
            assert_eq!(entry, CudaMetric::unavailable());
        }
        assert!(metric.value().is_none());
        let snapshot = metric.snapshot(true);
        assert!(
            snapshot.current.is_none()
                && snapshot.running.is_none()
                && snapshot.final_value.is_none()
        );
        assert_eq!(metric.clone().serialized_final(), entry);
        metric.clear();
        assert_eq!(metric.serialized_update(), entry);
        println!("SYSTEM_CUDA {}", entry.formatted);
    }

    #[test]
    fn system_metrics_integrate_without_changing_training_control() {
        use crate::{EventProcessorTraining, LearnerEvent, MetricsProcessor, Split, TrainingItem};
        let mut processor = MetricsProcessor::<(), ()>::new()
            .metric_train(CpuUse::new())
            .metric_train(CpuMemory::new())
            .metric_train(CpuTemperature::new())
            .metric_train(CudaMetric::new());
        processor
            .process_train(LearnerEvent::StartSplit {
                epoch: 1,
                total_items: 1,
            })
            .unwrap();
        processor
            .process_train(LearnerEvent::ProcessedItem(TrainingItem {
                item: (),
                iteration: 1,
                learning_rate: None,
                progress: ruda_model::data::dataloader::Progress {
                    items_processed: 1,
                    items_total: 1,
                },
            }))
            .unwrap();
        processor
            .process_train(LearnerEvent::EndSplit {
                epoch: 1,
                completed: true,
            })
            .unwrap();
        assert_eq!(processor.store().metrics(1, Split::Train).len(), 4);
        let snapshots = processor.metric_snapshots(Split::Train);
        assert!(snapshots[3].final_value.is_none());
        assert_eq!(
            snapshots[0].serialized.serialized.parse::<f64>().unwrap(),
            format!("{:.2}", snapshots[0].current.as_ref().unwrap().current())
                .parse::<f64>()
                .unwrap()
        );
    }
}
