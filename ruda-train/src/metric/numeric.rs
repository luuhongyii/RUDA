use super::Metric;
use std::sync::Arc;

pub type MetricName = Arc<String>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MetricId {
    id: MetricName,
}
impl MetricId {
    pub fn new(id: MetricName) -> Self {
        Self { id }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricAttributes {
    Numeric(NumericAttributes),
    None,
}
#[derive(Clone, Debug, PartialEq)]
pub struct NumericAttributes {
    pub unit: Option<String>,
    pub higher_is_better: bool,
}
impl Default for NumericAttributes {
    fn default() -> Self {
        Self {
            unit: None,
            higher_is_better: true,
        }
    }
}
impl From<NumericAttributes> for MetricAttributes {
    fn from(value: NumericAttributes) -> Self {
        Self::Numeric(value)
    }
}

#[derive(Clone, Debug)]
pub struct MetricDefinition {
    pub metric_id: MetricId,
    pub name: MetricName,
    pub description: Option<String>,
    pub attributes: MetricAttributes,
}
impl MetricDefinition {
    pub fn new<M: Metric + ?Sized>(metric_id: MetricId, metric: &M) -> Self {
        Self {
            metric_id,
            name: Arc::new(metric.name().into()),
            description: metric.description(),
            attributes: metric.attributes(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericEntry {
    Value(f64),
    Aggregated { aggregated_value: f64, count: usize },
    Final(f64),
}
impl NumericEntry {
    pub fn current(&self) -> f64 {
        match self {
            Self::Value(value) | Self::Final(value) => *value,
            Self::Aggregated {
                aggregated_value, ..
            } => *aggregated_value,
        }
    }
    pub fn serialize(&self) -> String {
        match self {
            Self::Value(value) => value.to_string(),
            Self::Final(value) => format!("{value},final"),
            Self::Aggregated {
                aggregated_value,
                count,
            } => format!("{aggregated_value},{count}"),
        }
    }
    pub fn deserialize(entry: &str) -> Result<Self, String> {
        let values: Vec<_> = entry.split(',').collect();
        match values.as_slice() {
            [value] => value
                .parse::<f64>()
                .map(Self::Value)
                .map_err(|e| e.to_string()),
            [value, "final"] => value
                .parse::<f64>()
                .map(Self::Final)
                .map_err(|e| e.to_string()),
            [value, count] => Ok(Self::Aggregated {
                aggregated_value: value.parse::<f64>().map_err(|e| e.to_string())?,
                count: count.parse::<usize>().map_err(|e| e.to_string())?,
            }),
            _ => Err("Invalid number of values for numeric entry".into()),
        }
    }
    pub fn better_than(&self, other: &Self, higher_is_better: bool) -> bool {
        (self.current() > other.current()) == higher_is_better
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SerializedEntry {
    pub formatted: String,
    pub serialized: String,
}
impl SerializedEntry {
    pub const NOT_AVAILABLE: &'static str = "N/A";
    pub fn new(formatted: String, serialized: String) -> Self {
        Self {
            formatted,
            serialized,
        }
    }
    pub fn not_available(unit: Option<&str>) -> Self {
        let formatted = match unit {
            Some(unit) => format!("epoch N/A {unit} - batch N/A {unit}"),
            None => "epoch N/A - batch N/A".into(),
        };
        Self::new(formatted, Self::NOT_AVAILABLE.into())
    }
    pub fn is_not_available(&self) -> bool {
        self.serialized == Self::NOT_AVAILABLE || self.serialized.is_empty()
    }
}

pub struct FormatOptions {
    name: MetricName,
    unit: Option<String>,
    precision: Option<usize>,
}
impl FormatOptions {
    pub fn new(name: MetricName) -> Self {
        Self {
            name,
            unit: None,
            precision: None,
        }
    }
    pub fn unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.into());
        self
    }
    pub fn precision(mut self, precision: usize) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn name(&self) -> &MetricName {
        &self.name
    }
    pub fn unit_value(&self) -> &Option<String> {
        &self.unit
    }
    pub fn precision_value(&self) -> Option<usize> {
        self.precision
    }
}

pub fn format_float(value: f64, precision: usize) -> String {
    if 0.1_f64.powf(precision as f64 - 1.) >= value {
        format!("{value:.precision$e}")
    } else {
        format!("{value:.precision$}")
    }
}

pub(super) fn serialize_values(
    current: Option<NumericEntry>,
    running: Option<NumericEntry>,
    final_entry: Option<NumericEntry>,
    format: FormatOptions,
) -> SerializedEntry {
    let display = |entry: &Option<NumericEntry>| match entry {
        None => SerializedEntry::NOT_AVAILABLE.into(),
        Some(entry) => match format.precision {
            Some(precision) => format_float(entry.current(), precision),
            None => entry.current().to_string(),
        },
    };
    let batch = display(&current);
    let epoch = display(&running);
    let formatted = match format.unit {
        Some(unit) => format!("epoch {epoch} {unit} - batch {batch} {unit}"),
        None => format!("epoch {epoch} - batch {batch}"),
    };
    let serialized = final_entry
        .or(current)
        .or(running)
        .map(|entry| entry.serialize())
        .unwrap_or_else(|| SerializedEntry::NOT_AVAILABLE.into());
    SerializedEntry::new(formatted, serialized)
}

#[derive(Clone, Debug)]
pub struct MetricSnapshot {
    pub definition: MetricDefinition,
    pub current: Option<NumericEntry>,
    pub running: Option<NumericEntry>,
    pub final_value: Option<NumericEntry>,
    pub serialized: SerializedEntry,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NumericMetricState {
    sum: f64,
    weight: usize,
    current: f64,
    current_weight: usize,
}
impl Default for NumericMetricState {
    fn default() -> Self {
        Self::new()
    }
}
impl NumericMetricState {
    pub fn new() -> Self {
        Self {
            sum: 0.,
            weight: 0,
            current: f64::NAN,
            current_weight: 0,
        }
    }
    pub fn update(&mut self, value: f64, weight: usize) {
        self.current = value;
        self.current_weight = weight;
        if weight != 0 {
            self.sum += value * weight as f64;
            self.weight += weight;
        }
    }
    pub fn value(&self) -> Option<f64> {
        (self.weight != 0).then(|| self.sum / self.weight as f64)
    }
    pub fn current_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: self.current,
            count: self.current_weight,
        }
    }
    pub fn running_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: self.sum / self.weight as f64,
            count: self.weight,
        }
    }
    pub fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.sum / self.weight as f64)
    }
    pub fn compute_update(&self, format: FormatOptions) -> SerializedEntry {
        serialize_values(
            Some(self.current_value()),
            Some(self.running_value()),
            None,
            format,
        )
    }
    pub fn compute_final(&self, format: FormatOptions) -> SerializedEntry {
        serialize_values(
            Some(self.current_value()),
            Some(self.running_value()),
            Some(self.final_value()),
            format,
        )
    }
    pub fn clear(&mut self) {
        *self = Self::new();
    }
    pub fn reset(&mut self) {
        self.clear();
    }
}

macro_rules! numeric_attributes {
    ($unit:expr, $higher:expr, $precision:expr) => {
        fn attributes(&self) -> $crate::metric::MetricAttributes {
            $crate::metric::NumericAttributes {
                unit: ($unit as Option<&str>).map(str::to_owned),
                higher_is_better: $higher,
            }
            .into()
        }
        fn format_options(&self) -> $crate::metric::FormatOptions {
            let format =
                $crate::metric::FormatOptions::new(std::sync::Arc::new(self.name().into()))
                    .precision($precision);
            match $unit as Option<&str> {
                Some(unit) => format.unit(unit),
                None => format,
            }
        }
    };
}
macro_rules! numeric_state_metric {
    ($unit:expr, $higher:expr, $precision:expr) => {
        fn checkpoint(&mut self) -> Result<Vec<u8>, $crate::TrainingError> {
            $crate::continuation::encode(&self.state)
        }
        fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), $crate::TrainingError> {
            self.state = $crate::continuation::decode(bytes)?;
            Ok(())
        }
        numeric_attributes!($unit, $higher, $precision);
        fn current_value(&self) -> Option<$crate::metric::NumericEntry> {
            Some(self.state.current_value())
        }
        fn running_value(&self) -> Option<$crate::metric::NumericEntry> {
            Some(self.state.running_value())
        }
        fn final_value(&self) -> $crate::metric::NumericEntry {
            self.state.final_value()
        }
    };
}
