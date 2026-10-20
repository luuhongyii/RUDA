/// Calibration method used to compute the quantization range mapping.
pub enum Calibration {
    /// Computes quantization range mapping based on the min and max values.
    MinMax,
    /// Exact interpolated quantiles in [0,1], independently for every block.
    Percentile { lower: f64, upper: f64 },
    /// Search symmetric integer clipping ratios 1, 1-1/steps, ..., 1/steps.
    Mse { steps: usize },
}
