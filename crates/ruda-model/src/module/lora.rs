use super::{Module, Param, ParamGroup, Quantizer, Reparameterization, Reparameterizer};
use ruda_tensor::{Backend, api::Tensor, Distribution, FloatDType};

/// Low-rank state for a structural matrix parameter.
#[derive(Module, Debug)]
pub struct LoraAdapter<B: Backend> {
    /// Trainable input-to-rank factor.
    pub a: Param<Tensor<B, 2>>,
    /// Trainable rank-to-output factor.
    pub b: Param<Tensor<B, 2>>,
    /// Multiplier of the low-rank product.
    pub scale: f64,
}

impl<B: Backend> LoraAdapter<B> {
    /// Compute the current trainable low-rank update.
    pub fn delta(&self) -> Tensor<B, 2> { self.a.val().matmul(self.b.val()).mul_scalar(self.scale) }
}

impl<B: Backend> Reparameterization<B> for LoraAdapter<B> {
    const NAME: &'static str = "lora";
    fn apply<const D: usize>(&self, base: Tensor<B, D>) -> Tensor<B, D> {
        let delta = self.delta().reshape(base.shape());
        let base = if base.dtype().is_float() && base.dtype() != delta.dtype() {
            base.cast(FloatDType::from(delta.dtype()))
        } else { base };
        base + delta
    }
}

/// Model-wide LoRA selection and initialization.
#[derive(Debug, Clone)]
pub struct Lora {
    /// Rank of the trainable product.
    pub rank: usize,
    /// Scaling numerator (`alpha / rank`).
    pub alpha: f64,
    /// Standard deviation of A; defaults to `1 / rank`.
    pub init_std: Option<f64>,
    /// Factor precision; defaults to the dense base's precision or the backend default.
    pub dtype: Option<FloatDType>,
    /// Parameters eligible for adapters. All floating bases are frozen.
    pub param_group: ParamGroup,
}

impl Lora {
    /// Create a LoRA configuration.
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self { rank, alpha, init_std: None, dtype: None, param_group: ParamGroup::all() }
    }
    /// Select parameters by ID or path.
    pub fn set_param_group(mut self, group: ParamGroup) -> Self { self.param_group = group; self }
    /// Set the trainable factors' floating precision.
    pub fn set_dtype(mut self, dtype: FloatDType) -> Self { self.dtype = Some(dtype); self }
}

impl<B: Backend> Reparameterizer<B> for Lora {
    type Reparam = LoraAdapter<B>;
    fn reparameterize<const D: usize>(&mut self, path: &str, param: Param<Tensor<B, D>>)
        -> (Param<Tensor<B, D>>, Option<Self::Reparam>)
    {
        if D != 2 || !self.param_group.matches(&param.id, Some(path)) {
            return (param.set_require_grad(false), None);
        }
        let base = param.base();
        let device = base.device();
        let context = base.autodiff_context();
        let dims = base.dims();
        let dtype = self.dtype.or_else(|| base.dtype().is_float().then(|| FloatDType::from(base.dtype())));
        let rank = self.rank;
        let std = self.init_std.unwrap_or(1.0 / rank as f64);
        let (a, b) = B::memory_persistent_allocations(&device, (), |_| {
            let a = Tensor::<B, 2>::random([dims[0], rank], Distribution::Normal(0.0, std), &device)
                .with_autodiff_context(context);
            let b = Tensor::<B, 2>::zeros([rank, dims[1]], &device).with_autodiff_context(context);
            match dtype { Some(dtype) => (a.cast(dtype), b.cast(dtype)), None => (a, b) }
        });
        (param.set_require_grad(false), Some(LoraAdapter {
            a: Param::from_tensor(a).set_require_grad(true),
            b: Param::from_tensor(b).set_require_grad(true), scale: self.alpha / rank as f64,
        }))
    }
}

/// Quantized frozen bases with independently trainable LoRA factors.
pub struct QLora {
    lora: Lora,
    quantizer: Quantizer,
    quantized_group: ParamGroup,
}

impl QLora {
    /// Combine adapter initialization with a quantization policy.
    pub fn new(lora: Lora, quantizer: Quantizer) -> Self { Self { lora, quantizer, quantized_group: ParamGroup::all() } }
    /// Select quantized bases independently of the LoRA adapter selection.
    pub fn set_quantized_group(mut self, group: ParamGroup) -> Self { self.quantized_group = group; self }
}

impl<B: Backend> Reparameterizer<B> for QLora {
    type Reparam = LoraAdapter<B>;
    fn reparameterize<const D: usize>(&mut self, path: &str, param: Param<Tensor<B, D>>)
        -> (Param<Tensor<B, D>>, Option<Self::Reparam>)
    {
        let param = self.quantizer.map_float_at_path(param, path, &self.quantized_group);
        self.lora.reparameterize(path, param)
    }
}
