use ruda_tensor::api::{
    Tensor,
    backend::Backend,
    quantization::{Calibration, QuantScheme, compute_q_params, compute_range},
};

use crate::module::{ModuleMapper, Param, ParamGroup};
use alloc::{string::{String, ToString}, vec::Vec};

/// Describes how to quantize a module.
pub struct Quantizer {
    /// The calibration method used in quantization.
    pub calibration: Calibration,
    /// The quantization scheme.
    pub scheme: QuantScheme,
}

impl Quantizer {
    /// Construct a quantizer without changing the existing literal configuration format.
    pub fn new(calibration: Calibration, scheme: QuantScheme) -> Self { Self { calibration, scheme } }

    /// Restrict quantization by parameter ID or module path.
    pub fn with_param_group(self, group: ParamGroup) -> GroupedQuantizer {
        GroupedQuantizer { quantizer: self, group, path: Vec::new() }
    }

    pub(crate) fn map_float_at_path<B: Backend, const D: usize>(&self, param: Param<Tensor<B, D>>, path: &str, group: &ParamGroup)
        -> Param<Tensor<B, D>>
    {
        if !group.matches(&param.id, Some(path)) { return param; }
        param.map(|tensor| {
            let range = compute_range(&self.scheme, &tensor, &self.calibration);
            let qparams = compute_q_params(&self.scheme, range);
            tensor.quantize(&self.scheme, qparams)
        })
    }
}

/// A path-aware quantizer for a selected parameter group.
pub struct GroupedQuantizer {
    /// Calibration and quantization scheme.
    pub quantizer: Quantizer,
    /// The selected parameter group.
    pub group: ParamGroup,
    path: Vec<String>,
}

impl<B: Backend> ModuleMapper<B> for GroupedQuantizer {
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        self.quantizer.map_float_at_path(param, &self.path.join("."), &self.group)
    }
}

impl<B: Backend> ModuleMapper<B> for Quantizer {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let range = compute_range(&self.scheme, &tensor, &self.calibration);
        let qparams = compute_q_params(&self.scheme, range);
        let tensor = tensor.quantize(&self.scheme, qparams);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(all(test, not(feature = "test-tch")))]
mod tests {
    use crate::test_utils::SimpleLinear;
    use crate::{
        TestBackend,
        module::{Module, Quantizer},
    };
    use ruda_tensor::api::{
        Device, Tolerance,
        ops::QuantizedTensor,
        quantization::{Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantValue},
    };

    type B = TestBackend;

    #[test]
    fn should_quantize_module() {
        let device: Device<B> = Default::default();
        let module = SimpleLinear::<B>::new(32, 32, &device);
        let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32);

        let result = module.weight.val();

        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };
        let q_module = module.quantize_weights(&mut quantizer);
        let q_result = q_module.weight.val().dequantize();

        result
            .into_data()
            .assert_approx_eq::<f32>(&q_result.into_data(), Tolerance::permissive());
    }
}
