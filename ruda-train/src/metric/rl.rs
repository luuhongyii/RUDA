use super::{
    FormatOptions, Metric, MetricAttributes, NumericAttributes, NumericEntry, NumericMetricState,
};
use crate::TrainingError;
use std::sync::Arc;

macro_rules! episode_metric {
    ($metric:ident, $input:ident, $field:ident, $name:literal, $unit:expr, $higher:expr, $precision:expr) => {
        pub struct $input {
            $field: f64,
        }
        impl $input {
            pub fn new($field: f64) -> Self {
                Self { $field }
            }
        }
        #[derive(Clone, Default)]
        pub struct $metric {
            state: NumericMetricState,
        }
        impl $metric {
            pub fn new() -> Self {
                Self::default()
            }
        }
        impl Metric for $metric {
            type Input = $input;
            fn name(&self) -> &str {
                $name
            }
            fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
                self.state.update(input.$field, 1);
                Ok(())
            }
            fn value(&self) -> Option<f64> {
                self.state.value()
            }
            fn current_value(&self) -> Option<NumericEntry> {
                Some(self.state.current_value())
            }
            fn running_value(&self) -> Option<NumericEntry> {
                Some(self.state.running_value())
            }
            fn final_value(&self) -> NumericEntry {
                self.state.final_value()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: ($unit as Option<&str>).map(str::to_owned),
                    higher_is_better: $higher,
                }
                .into()
            }
            fn format_options(&self) -> FormatOptions {
                FormatOptions::new(Arc::new(self.name().into())).precision($precision)
            }
            fn clear(&mut self) {
                self.state.reset();
            }
        }
    };
}

episode_metric!(
    CumulativeRewardMetric,
    CumulativeRewardInput,
    cum_reward,
    "Cum. Reward",
    None,
    true,
    2
);
episode_metric!(
    EpisodeLengthMetric,
    EpisodeLengthInput,
    ep_len,
    "Episode length",
    Some("steps"),
    true,
    0
);
episode_metric!(
    ExplorationRateMetric,
    ExplorationRateInput,
    exploration_rate,
    "Exploration rate",
    Some("%"),
    false,
    3
);

#[cfg(test)]
mod tests {
    use super::*;

    fn check<M: Metric + Clone>(
        mut metric: M,
        first: M::Input,
        second: M::Input,
        expected_first: f64,
        expected_second: f64,
        expected_unit: Option<&str>,
        higher: bool,
        expected_format: &str,
    ) {
        assert!(metric.value().is_none());
        metric.update(&first).unwrap();
        assert_eq!(metric.current_value().unwrap().current(), expected_first);
        let cloned = metric.clone();
        metric.update(&second).unwrap();
        let mean = (expected_first + expected_second) / 2.;
        assert_eq!(
            metric.current_value(),
            Some(NumericEntry::Aggregated {
                aggregated_value: expected_second,
                count: 1,
            })
        );
        assert_eq!(
            metric.running_value(),
            Some(NumericEntry::Aggregated {
                aggregated_value: mean,
                count: 2,
            })
        );
        assert_eq!(cloned.value(), Some(expected_first));
        assert_eq!(metric.value(), Some(mean));
        metric.compute().unwrap();
        assert_eq!(metric.final_value(), NumericEntry::Final(mean));
        assert_eq!(
            metric.serialized_final().serialized,
            format!("{mean},final")
        );
        assert_eq!(metric.serialized_update().formatted, expected_format);
        assert_eq!(
            metric.attributes(),
            NumericAttributes {
                unit: expected_unit.map(str::to_owned),
                higher_is_better: higher,
            }
            .into()
        );
        metric.clear();
        assert!(metric.value().is_none());
        assert!(metric.current_value().unwrap().current().is_nan());
    }

    #[test]
    fn cumulative_reward_preserves_negative_values_and_episode_weight() {
        check(
            CumulativeRewardMetric::new(),
            CumulativeRewardInput::new(-4.),
            CumulativeRewardInput::new(10.),
            -4.,
            10.,
            None,
            true,
            "epoch 3.00 - batch 10.00",
        );
    }

    #[test]
    fn episode_length_is_not_a_weight_for_the_metric_average() {
        check(
            EpisodeLengthMetric::new(),
            EpisodeLengthInput::new(20.),
            EpisodeLengthInput::new(100.),
            20.,
            100.,
            Some("steps"),
            true,
            "epoch 60 - batch 100",
        );
    }

    #[test]
    fn exploration_rate_keeps_input_scale_and_baseline_format() {
        check(
            ExplorationRateMetric::new(),
            ExplorationRateInput::new(0.5),
            ExplorationRateInput::new(0.125),
            0.5,
            0.125,
            Some("%"),
            false,
            "epoch 0.312 - batch 0.125",
        );
    }

    struct Episode {
        reward: f64,
        length: f64,
        exploration: f64,
    }
    impl super::super::Adaptor<CumulativeRewardInput> for Episode {
        fn adapt(&self) -> CumulativeRewardInput {
            CumulativeRewardInput::new(self.reward)
        }
    }
    impl super::super::Adaptor<EpisodeLengthInput> for Episode {
        fn adapt(&self) -> EpisodeLengthInput {
            EpisodeLengthInput::new(self.length)
        }
    }
    impl super::super::Adaptor<ExplorationRateInput> for Episode {
        fn adapt(&self) -> ExplorationRateInput {
            ExplorationRateInput::new(self.exploration)
        }
    }

    #[test]
    fn episode_metrics_integrate_with_processor_and_completed_split_store() {
        use crate::{EventProcessorTraining, LearnerEvent, MetricsProcessor, Split, TrainingItem};
        let mut processor = MetricsProcessor::<Episode, Episode>::new()
            .metric_train(CumulativeRewardMetric::new())
            .metric_train(EpisodeLengthMetric::new())
            .metric_train(ExplorationRateMetric::new());
        processor
            .process_train(LearnerEvent::StartSplit {
                epoch: 1,
                total_items: 2,
            })
            .unwrap();
        for (index, item) in [
            Episode {
                reward: -2.,
                length: 10.,
                exploration: 0.5,
            },
            Episode {
                reward: 6.,
                length: 30.,
                exploration: 0.25,
            },
        ]
        .into_iter()
        .enumerate()
        {
            processor
                .process_train(LearnerEvent::ProcessedItem(TrainingItem {
                    item,
                    iteration: index + 1,
                    learning_rate: None,
                    progress: ruda_model::data::dataloader::Progress {
                        items_processed: index + 1,
                        items_total: 2,
                    },
                }))
                .unwrap();
        }
        assert!(
            processor
                .metric_snapshots(Split::Train)
                .iter()
                .all(|m| m.final_value.is_none())
        );
        processor
            .process_train(LearnerEvent::EndSplit {
                epoch: 1,
                completed: true,
            })
            .unwrap();
        let store = processor.store();
        assert_eq!(store.find_metric("Cum. Reward", 1, Split::Train), Some(2.));
        assert_eq!(
            store.find_metric("Episode length", 1, Split::Train),
            Some(20.)
        );
        assert_eq!(
            store.find_metric("Exploration rate", 1, Split::Train),
            Some(0.375)
        );
        assert!(
            processor
                .metric_snapshots(Split::Train)
                .iter()
                .all(|m| matches!(m.final_value, Some(NumericEntry::Final(_))))
        );
    }
}
