#[derive(Clone, Debug)]
pub(crate) struct DisplayProgress {
    pub items_processed: usize,
    pub items_total: usize,
    pub unit: Option<String>,
}

impl DisplayProgress {
    pub fn new(items_processed: usize, items_total: usize, unit: Option<String>) -> Self {
        Self {
            items_processed,
            items_total,
            unit,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DisplayProgressSnapshot {
    pub global: DisplayProgress,
    pub split: DisplayProgress,
}

impl DisplayProgressSnapshot {
    pub fn new(global: DisplayProgress, split: DisplayProgress) -> Self {
        Self { global, split }
    }
}
