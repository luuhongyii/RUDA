use crate::{
    Split,
    metric::{EventStore, NumericEntry},
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    fmt,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq)]
pub struct MetricEntry {
    pub step: usize,
    pub value: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MetricSummary {
    pub name: String,
    pub entries: Vec<MetricEntry>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SummaryMetrics {
    pub train: Vec<MetricSummary>,
    pub valid: Vec<MetricSummary>,
    pub test: HashMap<String, Vec<MetricSummary>>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LearnerSummary {
    pub epochs: usize,
    pub metrics: SummaryMetrics,
    pub(crate) model: Option<String>,
}

impl LearnerSummary {
    pub fn new<S: AsRef<str>>(directory: impl AsRef<Path>, metrics: &[S]) -> Result<Self, String> {
        let directory = directory.as_ref();
        if !directory.exists() {
            return Err(format!(
                "Artifact directory does not exist at: {}",
                directory.display()
            ));
        }
        if !["train", "valid", "test"]
            .iter()
            .any(|split| directory.join(split).is_dir())
        {
            return Err(format!(
                "No training, validation or test artifacts found at: {}",
                directory.display()
            ));
        }
        let train = epoch_directories(&directory.join("train"))?;
        let valid = epoch_directories(&directory.join("valid"))?;
        let mut test = BTreeMap::new();
        let untagged = epoch_directories(&directory.join("test"))?;
        if !untagged.is_empty() {
            test.insert(String::new(), untagged);
        }
        if directory.join("test").is_dir() {
            for entry in std::fs::read_dir(directory.join("test")).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let tag = entry.file_name().to_string_lossy().into_owned();
                if entry.file_type().map_err(|e| e.to_string())?.is_dir()
                    && !tag.starts_with("epoch-")
                {
                    test.insert(tag, epoch_directories(&entry.path())?);
                }
            }
        }
        let epochs = train
            .keys()
            .chain(valid.keys())
            .chain(test.values().flat_map(|dirs| dirs.keys()))
            .copied()
            .max()
            .unwrap_or(0);
        let mut test_summaries = HashMap::new();
        for (tag, dirs) in test {
            test_summaries.insert(tag, collect_files(&dirs, metrics)?);
        }
        Ok(Self {
            epochs,
            metrics: SummaryMetrics {
                train: collect_files(&train, metrics)?,
                valid: collect_files(&valid, metrics)?,
                test: test_summaries,
            },
            model: None,
        })
    }
    pub fn new_eval<S: AsRef<str>>(
        directory: impl AsRef<Path>,
        metrics: &[S],
    ) -> Result<Self, String> {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            return Err(format!(
                "Artifact directory does not exist at: {}",
                directory.display()
            ));
        }
        let dirs = BTreeMap::from([(1, directory.to_path_buf())]);
        Ok(Self {
            epochs: 1,
            metrics: SummaryMetrics {
                test: HashMap::from([(String::new(), collect_files(&dirs, metrics)?)]),
                ..Default::default()
            },
            model: None,
        })
    }
    pub fn from_store(store: &EventStore, epochs: usize) -> Self {
        let snapshot = store.snapshot();
        let collect = |split| {
            let mut values: BTreeMap<String, Vec<MetricEntry>> = BTreeMap::new();
            for ((epoch, entry_split), entries) in &snapshot {
                if *entry_split != split || *epoch > epochs {
                    continue;
                }
                for entry in entries {
                    if let Some(value) = entry.value {
                        values
                            .entry(entry.name.clone())
                            .or_default()
                            .push(MetricEntry {
                                step: *epoch,
                                value,
                            });
                    }
                }
            }
            values
                .into_iter()
                .map(|(name, entries)| MetricSummary { name, entries })
                .collect::<Vec<_>>()
        };
        let test = collect(Split::Test);
        let test = if test.is_empty() {
            HashMap::new()
        } else {
            HashMap::from([(String::new(), test)])
        };
        Self {
            epochs,
            metrics: SummaryMetrics {
                train: collect(Split::Train),
                valid: collect(Split::Valid),
                test,
            },
            model: None,
        }
    }
    pub fn from_named_evaluation_store<S: AsRef<str>>(store: &EventStore, names: &[S]) -> Self {
        let test = names
            .iter()
            .map(|name| {
                let name = name.as_ref();
                let metrics = store
                    .test_metrics(1, name)
                    .into_iter()
                    .filter_map(|entry| {
                        entry.value.map(|value| MetricSummary {
                            name: entry.name,
                            entries: vec![MetricEntry { step: 1, value }],
                        })
                    })
                    .collect();
                (name.into(), metrics)
            })
            .collect();
        Self {
            epochs: usize::from(!names.is_empty()),
            metrics: SummaryMetrics {
                test,
                ..Default::default()
            },
            model: None,
        }
    }
    pub fn new_named_eval<S: AsRef<str>, N: AsRef<str>>(
        directory: impl AsRef<Path>,
        metrics: &[S],
        names: &[N],
    ) -> Result<Self, String> {
        if !names.is_empty() && !directory.as_ref().is_dir() {
            return Err(format!(
                "Artifact directory does not exist at: {}",
                directory.as_ref().display()
            ));
        }
        let mut test = HashMap::new();
        for name in names {
            let path = directory
                .as_ref()
                .join("test")
                .join(crate::logger::format_tag(name.as_ref()));
            let dirs = BTreeMap::from([(1, path)]);
            test.insert(name.as_ref().to_owned(), collect_files(&dirs, metrics)?);
        }
        Ok(Self {
            epochs: usize::from(!names.is_empty()),
            metrics: SummaryMetrics {
                test,
                ..Default::default()
            },
            model: None,
        })
    }
    pub(crate) fn from_evaluation_store(
        store: &EventStore,
        epoch: usize,
        split: Split,
        name: &str,
    ) -> Self {
        let entries = store
            .metrics(epoch, split)
            .into_iter()
            .filter_map(|entry| {
                entry.value.map(|value| MetricSummary {
                    name: entry.name,
                    entries: vec![MetricEntry { step: epoch, value }],
                })
            })
            .collect();
        Self {
            epochs: epoch,
            metrics: SummaryMetrics {
                test: HashMap::from([(name.into(), entries)]),
                ..Default::default()
            },
            model: None,
        }
    }
    pub(crate) fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }
    pub(crate) fn through_epoch(mut self, epochs: usize) -> Self {
        fn retain(metrics: &mut Vec<MetricSummary>, epochs: usize) {
            for metric in metrics.iter_mut() {
                metric.entries.retain(|entry| entry.step <= epochs);
            }
            metrics.retain(|metric| !metric.entries.is_empty());
        }
        self.epochs = epochs;
        retain(&mut self.metrics.train, epochs);
        retain(&mut self.metrics.valid, epochs);
        for metrics in self.metrics.test.values_mut() {
            retain(metrics, epochs);
        }
        self
    }
    pub fn merge(mut self, other: Self) -> Self {
        fn merge_metrics(
            base: Vec<MetricSummary>,
            incoming: Vec<MetricSummary>,
        ) -> Vec<MetricSummary> {
            let mut values: HashMap<String, MetricSummary> = base
                .into_iter()
                .map(|metric| (metric.name.clone(), metric))
                .collect();
            for metric in incoming {
                match values.entry(metric.name.clone()) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().entries.extend(metric.entries)
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(metric);
                    }
                }
            }
            values.into_values().collect()
        }
        self.metrics.train = merge_metrics(self.metrics.train, other.metrics.train);
        self.metrics.valid = merge_metrics(self.metrics.valid, other.metrics.valid);
        for (tag, metrics) in other.metrics.test {
            let current = self.metrics.test.remove(&tag).unwrap_or_default();
            self.metrics
                .test
                .insert(tag, merge_metrics(current, metrics));
        }
        if self.model != other.model {
            self.model = None;
        }
        self
    }
}

fn epoch_directories(directory: &Path) -> Result<BTreeMap<usize, PathBuf>, String> {
    if !directory.is_dir() {
        return Ok(BTreeMap::new());
    }
    let mut result = BTreeMap::new();
    for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            continue;
        }
        if let Some(epoch) = entry
            .file_name()
            .to_string_lossy()
            .strip_prefix("epoch-")
            .and_then(|n| n.parse::<usize>().ok())
        {
            if epoch != 0 {
                result.insert(epoch, entry.path());
            }
        }
    }
    Ok(result)
}
fn collect_files<S: AsRef<str>>(
    directories: &BTreeMap<usize, PathBuf>,
    metrics: &[S],
) -> Result<Vec<MetricSummary>, String> {
    let mut result = Vec::new();
    for metric in metrics {
        let name = metric.as_ref();
        let mut entries = Vec::new();
        for (epoch, directory) in directories {
            let path = directory.join(format!("{}.log", name.replace(' ', "_")));
            let points = crate::logger::FileMetricLogger::read_numeric_file(&path)
                .map_err(|e| e.to_string())?;
            if let Some(value) = aggregate(&points)? {
                entries.push(MetricEntry {
                    step: *epoch,
                    value,
                });
            }
        }
        if !entries.is_empty() {
            result.push(MetricSummary {
                name: name.into(),
                entries,
            });
        }
    }
    Ok(result)
}
fn aggregate(points: &[NumericEntry]) -> Result<Option<f64>, String> {
    let Some(last) = points.last() else {
        return Ok(None);
    };
    if let NumericEntry::Final(value) = last {
        return Ok(Some(*value));
    }
    let (mut sum, mut count) = (0., 0usize);
    for point in points {
        match point {
            NumericEntry::Value(value) => {
                sum += value;
                count += 1;
            }
            NumericEntry::Aggregated {
                aggregated_value,
                count: n,
            } => {
                sum += aggregated_value * *n as f64;
                count += n;
            }
            NumericEntry::Final(_) => {
                return Err("Final numeric entry must be the last entry".into());
            }
        }
    }
    Ok(Some(sum / count as f64))
}

impl fmt::Display for LearnerSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut split_width = 5;
        let mut metric_width = "Metric".len();
        for metric in self.metrics.train.iter().chain(&self.metrics.valid) {
            metric_width = metric_width.max(metric.name.len());
        }
        for (tag, metrics) in &self.metrics.test {
            let split = if tag.is_empty() {
                "Test".into()
            } else {
                format!("Test ({tag})")
            };
            split_width = split_width.max(split.len());
            for metric in metrics {
                metric_width = metric_width.max(metric.name.len());
            }
        }
        writeln!(f, "{:=>24} Learner Summary {:=>24}", "", "")?;
        if let Some(model) = &self.model {
            writeln!(f, "Model:\n{model}")?;
        }
        writeln!(f, "Total Epochs: {}\n\n", self.epochs)?;
        writeln!(
            f,
            "| {:<split_width$} | {:<metric_width$} | Min.     | Epoch    | Max.     | Epoch    |\n|{:->split_width$}--|{:->metric_width$}--|----------|----------|----------|----------|",
            "Split", "Metric", "", ""
        )?;
        fn compare(a: f64, b: f64) -> Ordering {
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => a.partial_cmp(&b).unwrap(),
            }
        }
        fn value(v: f64) -> String {
            if v < 1e-2 {
                format!("{v:<9.3e}")
            } else {
                format!("{v:<9.3}")
            }
        }
        let mut write = |metrics: &[MetricSummary], split: &str| -> fmt::Result {
            for metric in metrics {
                let Some(min) = metric
                    .entries
                    .iter()
                    .min_by(|a, b| compare(a.value, b.value))
                else {
                    continue;
                };
                let max = metric
                    .entries
                    .iter()
                    .max_by(|a, b| compare(a.value, b.value))
                    .unwrap();
                writeln!(
                    f,
                    "| {:<split_width$} | {:<metric_width$} | {}| {:<9?}| {}| {:<9?}|",
                    split,
                    metric.name,
                    value(min.value),
                    min.step,
                    value(max.value),
                    max.step
                )?;
            }
            Ok(())
        };
        write(&self.metrics.train, "Train")?;
        write(&self.metrics.valid, "Valid")?;
        for (tag, metrics) in &self.metrics.test {
            write(
                metrics,
                &if tag.is_empty() {
                    "Test".into()
                } else {
                    format!("Test ({tag})")
                },
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LearnerSummaryConfig {
    directory: PathBuf,
    metrics: Vec<String>,
}
impl LearnerSummaryConfig {
    pub fn new<S: AsRef<str>>(directory: impl AsRef<Path>, metrics: &[S]) -> Self {
        Self {
            directory: directory.as_ref().into(),
            metrics: metrics.iter().map(|name| name.as_ref().into()).collect(),
        }
    }
    pub fn init(&self) -> Result<LearnerSummary, String> {
        LearnerSummary::new(&self.directory, &self.metrics)
    }
    pub fn init_eval(&self) -> Result<LearnerSummary, String> {
        LearnerSummary::new_eval(&self.directory, &self.metrics)
    }
    pub fn init_named_eval<S: AsRef<str>>(&self, names: &[S]) -> Result<LearnerSummary, String> {
        LearnerSummary::new_named_eval(&self.directory, &self.metrics, names)
    }
}
