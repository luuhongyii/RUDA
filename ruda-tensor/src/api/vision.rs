//! Backend-generic non-maximum suppression.

use alloc::{vec, vec::Vec};
use super::{backend::Backend, DType, Int, Tensor, TensorData};

/// Thresholds and output limit for non-maximum suppression.
#[derive(Clone, Copy, Debug)]
pub struct NmsOptions {
    /// Suppress boxes whose intersection over union exceeds this threshold.
    pub iou_threshold: f32,
    /// Discard scores strictly below this threshold.
    pub score_threshold: f32,
    /// Maximum retained boxes; zero means unlimited.
    pub max_output_boxes: usize,
}

impl Default for NmsOptions {
    fn default() -> Self {
        Self { iou_threshold: 0.5, score_threshold: 0.0, max_output_boxes: 0 }
    }
}

/// Suppression of `[x1, y1, x2, y2]` boxes in descending score order.
pub trait Nms<B: Backend> {
    /// Return retained indices. Equal scores retain original index order.
    fn nms(self, scores: Tensor<B, 1>, options: NmsOptions) -> Tensor<B, 1, Int>;
}

impl<B: Backend> Nms<B> for Tensor<B, 2> {
    fn nms(self, scores: Tensor<B, 1>, options: NmsOptions) -> Tensor<B, 1, Int> {
        let context = self.autodiff_context().merge(scores.autodiff_context());
        let [count, coordinates] = self.dims();
        assert_eq!(coordinates, 4, "NMS requires four coordinates per box");
        assert_eq!(scores.dims(), [count], "NMS requires one score per box");
        let device = self.device();
        assert_eq!(scores.device(), device, "NMS inputs must share a device");
        assert!(!matches!(self.dtype(), DType::QFloat(_)), "NMS does not support quantized boxes");
        if count == 0 { return Tensor::<B, 1, Int>::zeros([0], &device).with_autodiff_context(context); }
        let boxes = self.into_data().convert::<f32>().into_vec::<f32>().expect("NMS boxes");
        let scores = scores.into_data().convert::<f32>().into_vec::<f32>().expect("NMS scores");
        let mut order: Vec<_> = (0..count).filter(|&i| scores[i] >= options.score_threshold).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let mut suppressed = vec![false; count];
        let mut keep = Vec::new();
        for (position, &index) in order.iter().enumerate() {
            if suppressed[index] { continue; }
            keep.push(index as i64);
            if options.max_output_boxes > 0 && keep.len() >= options.max_output_boxes { break; }
            let a = &boxes[index * 4..index * 4 + 4];
            let area_a = (a[2] - a[0]) * (a[3] - a[1]);
            for &other in &order[position + 1..] {
                if suppressed[other] { continue; }
                let b = &boxes[other * 4..other * 4 + 4];
                let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
                let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
                let intersection = width * height;
                let union = area_a + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
                suppressed[other] = intersection / union > options.iou_threshold;
            }
        }
        let count = keep.len();
        Tensor::<B, 1, Int>::from_data(TensorData::new(keep, [count]), &device).with_autodiff_context(context)
    }
}
