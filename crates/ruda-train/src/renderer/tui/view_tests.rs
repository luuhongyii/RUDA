use super::*;
use crate::metric::{
    MetricAttributes, MetricDefinition, MetricId, MetricSnapshot, NumericAttributes, NumericEntry,
    SerializedEntry,
};
use ratatui::{
    Terminal,
    backend::TestBackend,
    crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
};
use std::sync::Arc;

#[test]
fn full_recent_summary_and_named_evaluation_render_actual_widgets() {
    let name = Arc::new("Loss".to_string());
    let definition = MetricDefinition {
        metric_id: MetricId::new(name.clone()),
        name: name.clone(),
        description: None,
        attributes: MetricAttributes::Numeric(NumericAttributes::default()),
    };
    let mut numeric = NumericMetricsState::default();
    let mut text = TextMetricsState::default();
    for (split, group) in [
        (TuiSplit::Train, TuiGroup::Default),
        (TuiSplit::Valid, TuiGroup::Default),
        (TuiSplit::Test, TuiGroup::Named(Arc::new("holdout".into()))),
    ] {
        let tag = TuiTag::new(split, group.clone());
        numeric.push(tag.clone(), name.clone(), None);
        numeric.push(tag.clone(), name.clone(), Some(NumericEntry::Value(1.0)));
        numeric.push(tag, name.clone(), Some(NumericEntry::Final(0.75)));
        text.update(
            split,
            group,
            MetricSnapshot {
                definition: definition.clone(),
                current: Some(NumericEntry::Value(1.0)),
                running: Some(NumericEntry::Value(0.75)),
                final_value: Some(NumericEntry::Final(0.75)),
                serialized: SerializedEntry::new(
                    "epoch 0.75 - batch 1".into(),
                    "0.75,final".into(),
                ),
            },
            name.clone(),
        );
    }
    let snapshot = DisplayProgressSnapshot::new(
        DisplayProgress::new(1, 2, Some("epochs".into())),
        DisplayProgress::new(5, 10, Some("items".into())),
    );
    let mut progress = ProgressBarState::new(None);
    progress.update_train(&snapshot);
    let mut status = StatusState::default();
    status.update_label(Some("fit"));
    status.update_train(&snapshot);
    status.update_counter("Iteration".into());
    let mut terminal = Terminal::new(TestBackend::new(140, 36)).unwrap();
    for expected in ["Full History", "Recent History", "Summary"] {
        terminal
            .draw(|frame| {
                MetricsView::new(
                    numeric.view(),
                    text.view(),
                    progress.view(),
                    ControlsView,
                    status.view(),
                )
                .render(frame, frame.area());
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        let output: String = buffer.content.iter().map(|cell| cell.symbol()).collect();
        for label in [
            expected,
            "Loss",
            "holdout",
            "Progress",
            "Status: fit",
            "Iteration",
        ] {
            assert!(output.contains(label), "missing {label}: {output}");
        }
        numeric.on_event(&Event::Key(KeyEvent::new_with_kind(
            KeyCode::Down,
            KeyModifiers::NONE,
            KeyEventKind::Release,
        )));
    }
    terminal.backend_mut().resize(8, 4);
    terminal
        .draw(|frame| {
            MetricsView::new(
                numeric.view(),
                text.view(),
                progress.view(),
                ControlsView,
                status.view(),
            )
            .render(frame, frame.area());
        })
        .unwrap();
}
