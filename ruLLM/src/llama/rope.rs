use super::*;

#[derive(Clone, Debug)]
pub(crate) struct RopeParameters {
    pub(crate) theta: f32,
    pub(crate) inverse_frequencies: Vec<f32>,
    pub(crate) attention_factor: f64,
}
impl RopeParameters {
    pub(crate) fn default_for(config: &LlamaConfig) -> Self {
        let half = config.head_dimension() / 2;
        Self {theta: config.rope_theta, inverse_frequencies: (0..half)
            .map(|i| 1.0f32 / config.rope_theta.powf(i as f32 / half as f32)).collect(), attention_factor: 1.0}
    }
    pub(crate) fn init<B: Backend>(&self, config: &LlamaConfig, device: &B::Device, dtype: DType) -> RotaryEncoding<B> {
        let half = config.head_dimension() / 2;
        assert_eq!(self.inverse_frequencies.len(), half);
        let theta = Tensor::<B, 1>::from_data(TensorData::new(self.inverse_frequencies.clone(), [half]), (device, DType::F32));
        let mut rope = RotaryEncodingConfig::new(1, config.head_dimension()).with_theta(self.theta).init(device);
        rope.theta = theta.clone();
        let positions = Tensor::<B, 2>::from_data(TensorData::new(
            (0..config.max_sequence_length).map(|i| i as f32).collect::<Vec<_>>(), [config.max_sequence_length, 1]), (device, DType::F32));
        let frequencies = positions * theta.unsqueeze::<2>();
        let cos = frequencies.clone().cos() * self.attention_factor;
        let sin = frequencies.sin() * self.attention_factor;
        rope.freq_complex = Tensor::cat(vec![cos, sin], 1).reshape([config.max_sequence_length, 2, half])
            .transpose().unsqueeze_dim::<4>(2).repeat_dim(2, 2)
            .reshape([config.max_sequence_length, config.head_dimension(), 2]).cast(dtype);
        rope
    }
}

pub(super) fn qwen2_rope<B: Backend>(
    config: &LlamaConfig,
    device: &B::Device,
) -> RotaryEncoding<B> {
    let dtype = Tensor::<B, 1>::zeros([1], device).dtype();
    qwen2_rope_with_dtype(config, device, dtype)
}

pub(crate) fn qwen2_rope_with_dtype<B: Backend>(
    config: &LlamaConfig,
    device: &B::Device,
    dtype: DType,
) -> RotaryEncoding<B> {
    RopeParameters::default_for(config).init(config, device, dtype)
}
