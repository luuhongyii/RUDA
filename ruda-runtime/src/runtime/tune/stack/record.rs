use super::{TuneReport, TimeLimit, TuneFailure, cache::split_scoped_key, engine::SearchTrace};
use ruda_core::environment::records::{Record, RecordEffect, Span};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// The result of a shared-controller search in the active environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackTuneRecord {
    /// Versioned table containing the selected implementation.
    pub table: String,
    /// Full persistent key without the process-local generation prefix.
    pub entry: String,
    /// Candidate order supplied to the controller, including unmeasured candidates.
    pub plan: Vec<String>,
    /// Compilation, validation and measurement wall time for attempted candidates.
    pub trials: Vec<(String, Duration)>,
    /// Extra completed candidate samples used only to confirm a bound, never in paired scoring.
    #[serde(default)]
    pub confirmation_samples: Vec<(String, usize)>,
    /// The measured bound supplied by the trial adapter, when enabled.
    pub limit: Option<TimeLimit>,
    /// The selected candidate that satisfied the confirmation requirement.
    pub short_circuit: Option<String>,
    /// Per-candidate results and the selected implementation.
    pub report: Option<TuneReport>,
    /// Failure returned before a complete selection report was available.
    #[serde(default)]
    pub failure: Option<String>,
    /// Total recorded search wall time, including bound preparation.
    pub wall: Duration,
    /// Whether the cache table accepted the answer; not a claim of durable fsync.
    pub stored: bool,
    /// Whether a compile-only workload provoked this search.
    pub dry_run: bool,
}

impl Record for StackTuneRecord { const KIND: &'static str = "stack-autotune"; }

pub(super) struct StackRecording {
    span: Span,
    table: String,
    entry: String,
    plan: Vec<String>,
    dry_run: bool,
}

impl StackRecording {
    pub(super) fn new<'a>(table: &str, key: &str, names: impl Iterator<Item = &'a str>) -> Option<Self> {
        let (generation, entry) = split_scoped_key(key)?;
        if generation != ruda_core::environment::generation() { return None; }
        let span = Span::new()?;
        if generation != ruda_core::environment::generation() { return None; }
        Some(Self {
            span, table: table.into(),
            entry: entry.into(),
            plan: names.map(String::from).collect(),
            dry_run: crate::runtime::dry_run::dry_run(),
        })
    }

    pub(super) fn finish(self, report: &TuneReport, trace: &SearchTrace, limit: Option<TimeLimit>, stored: bool) {
        self.close(Some(report.clone()), None, trace, limit, stored);
    }

    pub(super) fn finish_failure(self, failure: &TuneFailure, trace: &SearchTrace, limit: Option<TimeLimit>) {
        self.close(None, Some(failure.to_string()), trace, limit, false);
    }

    fn close(self, report: Option<TuneReport>, failure: Option<String>, trace: &SearchTrace, limit: Option<TimeLimit>, stored: bool) {
        let Some(wall) = self.span.elapsed() else { return; };
        let record = StackTuneRecord {
            table: self.table, entry: self.entry, plan: self.plan, trials: trace.trials.clone(),
            confirmation_samples: trace.confirmation_samples.clone(),
            limit, short_circuit: trace.short_circuit.clone(), report, failure,
            wall, stored, dry_run: self.dry_run,
        };
        self.span.close(if stored { RecordEffect::Changed } else { RecordEffect::Observed }, &record);
    }
}
