void quantized_activation_input(const at::Tensor& self) {
  validate(self.device());
  TORCH_CHECK(self.is_quantized() && self.qscheme() == c10::kPerTensorAffine,
              "RUDA quantized activation requires per-tensor affine quantization");
  quantized_storage_type(self.scalar_type());
}

at::Tensor quantized_activation_empty(const at::Tensor& self, double scale, int64_t zero) {
  quantized_activation_input(self);
  return quantized_empty_affine(self.sizes(), self.scalar_type(), self.layout(), self.device(),
                               false, scale, zero, self.suggest_memory_format());
}

void quantized_activation_output(const at::Tensor& self, const at::Tensor& out) {
  quantized_activation_input(self); quantized_activation_input(out);
  TORCH_CHECK(out.device() == self.device() && out.scalar_type() == self.scalar_type(),
              "quantized activation input and output must have the same device and dtype");
  at::assert_no_internal_overlap(out);
  at::assert_no_partial_overlap(out, self);
  if (at::native::resize_output_check(out, self.sizes()))
    resize_tensor(out, self.sizes(), std::nullopt);
}

at::Tensor& quantized_activation_copy(const at::Tensor& value, at::Tensor& out) {
  quantized_activation_output(value, out);
  quantized_copy_from(value, out, false);
  return out;
}

void quantized_bounded_call(const at::Tensor& self, const at::Tensor& out, bool relu,
    const std::optional<at::Scalar>& lower, const std::optional<at::Scalar>& upper) {
  quantized_activation_output(self, out);
  TORCH_CHECK(relu || lower || upper, "At least one of 'min' or 'max' must not be None");
  TORCH_CHECK(self.q_scale() == out.q_scale() && self.q_zero_point() == out.q_zero_point(),
              "integer-domain activation output must share the input quantization parameters");
  scientific_call(552, {quantized_storage_view(self), quantized_storage_view(out)},
      {int64_t(!relu), self.q_zero_point(), int64_t(lower.has_value()), int64_t(upper.has_value())},
      {self.q_scale(), lower ? lower->toFloat() : 0.0, upper ? upper->toFloat() : 0.0});
}

void quantized_nonlinear_call(const at::Tensor& self, const at::Tensor& out, int64_t mode,
                            double p0 = 0.0, double p1 = 0.0, double p2 = 0.0) {
  quantized_activation_output(self, out);
  scientific_call(553, {quantized_storage_view(self), quantized_storage_view(out)},
      {mode, self.q_zero_point(), out.q_zero_point()},
      {self.q_scale(), out.q_scale(), p0, p1, p2});
}

at::Tensor quantized_nonlinear(const at::Tensor& self, int64_t mode, double scale, int64_t zero,
                             double p0 = 0.0, double p1 = 0.0, double p2 = 0.0) {
  auto out = quantized_activation_empty(self, scale, zero);
  quantized_nonlinear_call(self, out, mode, p0, p1, p2);
  return out;
}

at::Tensor quantized_relu(const at::Tensor& self) {
  auto out = quantized_activation_empty(self, self.q_scale(), self.q_zero_point());
  quantized_bounded_call(self, out, true, std::nullopt, std::nullopt);
  return out;
}
at::Tensor& quantized_relu_(at::Tensor& self) {
  quantized_bounded_call(self, self, true, std::nullopt, std::nullopt);
  return self;
}
at::Tensor& quantized_relu_out(const at::Tensor& self, at::Tensor& out) {
  return quantized_activation_copy(quantized_relu(self), out);
}

at::Tensor quantized_clamp(const at::Tensor& self, const std::optional<at::Scalar>& lower,
                          const std::optional<at::Scalar>& upper) {
  auto out = quantized_activation_empty(self, self.q_scale(), self.q_zero_point());
  quantized_bounded_call(self, out, false, lower, upper);
  return out;
}
at::Tensor quantized_hardtanh(const at::Tensor& self, const at::Scalar& lower, const at::Scalar& upper) {
  return quantized_clamp(self, lower, upper);
}
at::Tensor& quantized_hardtanh_(at::Tensor& self, const at::Scalar& lower, const at::Scalar& upper) {
  quantized_bounded_call(self, self, false, lower, upper);
  return self;
}
at::Tensor& quantized_hardtanh_out(const at::Tensor& self, const at::Scalar& lower,
                                  const at::Scalar& upper, at::Tensor& out) {
  out = quantized_clamp(self, lower, upper);
  return out;
}
at::Tensor quantized_relu6(at::Tensor self, bool inplace) {
  if (inplace) { quantized_hardtanh_(self, 0.0, 6.0); return self; }
  return quantized_hardtanh(self, 0.0, 6.0);
}

at::Tensor quantized_leaky_relu(const at::Tensor& self, const at::Scalar& slope) {
  return quantized_nonlinear(self, 0, self.q_scale(), self.q_zero_point(), slope.toFloat());
}
at::Tensor& quantized_leaky_relu_(at::Tensor& self, const at::Scalar& slope) {
  quantized_nonlinear_call(self, self, 0, slope.toFloat());
  return self;
}
at::Tensor& quantized_leaky_relu_out(const at::Tensor& self, const at::Scalar& slope, at::Tensor& out) {
  quantized_nonlinear_call(self, out, 0, slope.toFloat());
  return out;
}
at::Tensor quantized_leaky_relu_qparams(at::Tensor self, const at::Scalar& slope,
                                      bool inplace, double scale, int64_t zero) {
  if (inplace) TORCH_WARN("inplace=True is not supported for quantized::leaky_relu yet");
  return quantized_nonlinear(self, 0, scale, zero, slope.toFloat());
}

at::Tensor quantized_sigmoid_qparams(at::Tensor self, double scale, int64_t zero) {
  return quantized_nonlinear(self, 1, scale, zero);
}
at::Tensor quantized_sigmoid(const at::Tensor& self) {
  const double scale = self.scalar_type() == at::kQInt32 ? 0x1p-32 : 0x1p-8;
  const int64_t zero = self.scalar_type() == at::kQInt8 ? -128 : 0;
  return quantized_nonlinear(self, 1, scale, zero);
}
at::Tensor quantized_tanh(const at::Tensor& self) {
  const double scale = self.scalar_type() == at::kQInt32 ? 0x1p-31 : 0x1p-7;
  const int64_t zero = self.scalar_type() == at::kQUInt8 ? 128 : 0;
  return quantized_nonlinear(self, 2, scale, zero);
}
at::Tensor quantized_hardsigmoid(const at::Tensor& self) {
  const double scale = self.scalar_type() == at::kQInt32 ? 0x1p-32 : 0x1p-8;
  const int64_t zero = self.scalar_type() == at::kQInt8 ? -128 : 0;
  return quantized_nonlinear(self, 3, scale, zero);
}
at::Tensor& quantized_hardsigmoid_out(const at::Tensor& self, at::Tensor& out) {
  return quantized_activation_copy(quantized_hardsigmoid(self), out);
}
at::Tensor& quantized_hardsigmoid_(at::Tensor& self) {
  return quantized_hardsigmoid_out(self, self);
}

at::Tensor quantized_threshold(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  return quantized_nonlinear(self, 4, self.q_scale(), self.q_zero_point(), threshold.toFloat(), value.toFloat());
}
at::Tensor quantized_hardswish(const at::Tensor& self, double scale, int64_t zero) {
  return quantized_nonlinear(self, 5, scale, zero);
}
at::Tensor quantized_elu(const at::Tensor& self, double output_scale, int64_t zero,
                         const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {
  return quantized_nonlinear(self, 6, output_scale, zero, alpha.toFloat(), scale.toFloat(), input_scale.toFloat());
}
at::Tensor quantized_celu(const at::Tensor& self, double scale, int64_t zero, const at::Scalar& alpha) {
  TORCH_CHECK(alpha.toDouble() != 0, "ZeroDivisionError: alpha cannot be 0 for CELU");
  return quantized_elu(self, scale, zero, alpha, 1.0, 1.0 / alpha.toDouble());
}
at::Tensor quantized_gelu(const at::Tensor& self, std::string_view approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh");
  return quantized_nonlinear(self, approximate == "none" ? 7 : 8, self.q_scale(), self.q_zero_point());
}
at::Tensor& quantized_gelu_(at::Tensor& self, std::string_view approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh");
  quantized_nonlinear_call(self, self, approximate == "none" ? 7 : 8);
  return self;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("relu", TORCH_FN(quantized_relu));
  m.impl("relu_", TORCH_FN(quantized_relu_));
  m.impl("relu.out", TORCH_FN(quantized_relu_out));
  m.impl("clamp", TORCH_FN(quantized_clamp));
  m.impl("hardtanh", TORCH_FN(quantized_hardtanh));
  m.impl("hardtanh_", TORCH_FN(quantized_hardtanh_));
  m.impl("hardtanh.out", TORCH_FN(quantized_hardtanh_out));
  m.impl("leaky_relu", TORCH_FN(quantized_leaky_relu));
  m.impl("leaky_relu_", TORCH_FN(quantized_leaky_relu_));
  m.impl("leaky_relu.out", TORCH_FN(quantized_leaky_relu_out));
  m.impl("sigmoid", TORCH_FN(quantized_sigmoid));
  m.impl("tanh", TORCH_FN(quantized_tanh));
  m.impl("hardsigmoid", TORCH_FN(quantized_hardsigmoid));
  m.impl("hardsigmoid_", TORCH_FN(quantized_hardsigmoid_));
  m.impl("hardsigmoid.out", TORCH_FN(quantized_hardsigmoid_out));
  m.impl("threshold", TORCH_FN(quantized_threshold));
  m.impl("gelu", TORCH_FN(quantized_gelu));
  m.impl("gelu_", TORCH_FN(quantized_gelu_));
}
TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("relu6", TORCH_FN(quantized_relu6));
  m.impl("clamp", TORCH_FN(quantized_clamp));
  m.impl("leaky_relu", TORCH_FN(quantized_leaky_relu_qparams));
  m.impl("sigmoid", TORCH_FN(quantized_sigmoid_qparams));
  m.impl("threshold", TORCH_FN(quantized_threshold));
  m.impl("hardswish", TORCH_FN(quantized_hardswish));
  m.impl("elu", TORCH_FN(quantized_elu));
  m.impl("celu", TORCH_FN(quantized_celu));
}
