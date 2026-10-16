struct QuantizedReductionPlan {
  bool inner = false;
  int64_t rows = 1, width = 1;
  std::vector<int64_t> shape;
};

QuantizedReductionPlan quantized_reduction_plan(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim) {
  QuantizedReductionPlan plan;
  auto axes = dim.has_value() ? dim.value().vec() : std::vector<int64_t>{};
  if (axes.empty()) for (int64_t i = 0; i < self.dim(); ++i) axes.push_back(i);
  at::maybe_wrap_dims(axes, self.dim());
  std::sort(axes.begin(), axes.end());
  if (axes.size() > size_t(self.dim())) return plan;
  const auto first = self.dim() - int64_t(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) if (axes[i] != first + int64_t(i)) return plan;
  plan.inner = true; plan.shape = self.sizes().vec();
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (i < first) plan.rows *= self.size(i);
    else { plan.width *= self.size(i); plan.shape[i] = 1; }
  }
  if (!keepdim) plan.shape.erase(plan.shape.begin() + first, plan.shape.end());
  return plan;
}

at::Tensor quantized_inner_reduce(const at::Tensor& self, const QuantizedReductionPlan& plan,
    bool deviation, double correction) {
  auto output = quantized_empty_affine(plan.shape, self.scalar_type(), self.layout(), self.device(), false,
      self.q_scale(), self.q_zero_point(), self.suggest_memory_format());
  if (plan.rows == 0) return output;
  auto raw = quantized_storage_view(self).reshape({plan.rows, plan.width});
  const int64_t tiles = std::max<int64_t>(1, (plan.width + 1023) / 1024);
  auto sums = at::empty({plan.rows, tiles}, raw.options().dtype(at::kLong));
  auto squares = deviation ? at::empty({plan.rows, tiles}, raw.options().dtype(self.scalar_type() == at::kQInt32 ? at::kFloat : at::kLong)) :
      at::empty({0}, raw.options().dtype(self.scalar_type() == at::kQInt32 ? at::kFloat : at::kLong));
  scientific_call(574, {raw, sums, squares}, {int64_t(deviation)}, {});
  scientific_call(575, {sums, squares, quantized_storage_view(output).reshape({plan.rows})},
      {plan.width, self.q_zero_point(), int64_t(deviation)}, {self.q_scale(), correction});
  return output;
}

at::Tensor quantized_mean(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
    std::optional<at::ScalarType> dtype) {
  quantized_arithmetic_input(self);
  const auto plan = quantized_reduction_plan(self, dim, keepdim);
  if (self.is_contiguous() && plan.inner && (!dtype || *dtype == self.scalar_type()))
    return quantized_inner_reduce(self, plan, false, 0.0);
  auto real = at::native::dequantize_quantized(self);
  auto reduced = at::mean(real, dim, keepdim, dtype);
  return quantize_tensor_ruda(reduced, self.q_scale(), self.q_zero_point(), dtype.value_or(self.scalar_type()));
}

at::Tensor& quantized_mean_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
    std::optional<at::ScalarType> dtype, at::Tensor& result) {
  result = quantized_mean(self, dim, keepdim, dtype);
  return result;
}

at::Tensor quantized_std(const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<at::Scalar>& correction, bool keepdim) {
  quantized_arithmetic_input(self);
  const auto plan = quantized_reduction_plan(self, dim, keepdim);
  const double value = correction.value_or(at::Scalar(1)).toDouble();
  if (self.is_contiguous() && plan.inner && !(plan.width == 1 && !correction.value_or(at::Scalar(1)).equal(0)))
    return quantized_inner_reduce(self, plan, true, value);
  auto real = at::native::dequantize_quantized(self);
  auto reduced = at::std(real, dim, correction, keepdim);
  return quantize_tensor_ruda(reduced, self.q_scale(), self.q_zero_point(), self.scalar_type());
}

at::Tensor& quantized_std_out(const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<at::Scalar>& correction, bool keepdim, at::Tensor& result) {
  result = quantized_std(self, dim, correction, keepdim);
  return result;
}

at::Tensor quantized_channel_shuffle(const at::Tensor& self, c10::SymInt group_count) {
  quantized_arithmetic_input(self);
  const auto groups = group_count.expect_int();
  TORCH_CHECK(groups > 0, "channel_shuffle groups must be positive");
  TORCH_CHECK(self.dim() == 4, "quantized channel_shuffle expects4D input");
  TORCH_CHECK(self.scalar_type() == at::kQUInt8, "quantized channel_shuffle requires quint8");
  if (groups != 1) {
    TORCH_CHECK(self.size(1) > 0, "channel_shuffle channels must be positive");
    TORCH_CHECK(self.size(1) % groups == 0, "channel_shuffle channels must be divisible by groups");
  }
  auto result = quantized_empty_affine(self.sizes(), self.scalar_type(), self.layout(), self.device(), false,
      self.q_scale(), self.q_zero_point(), self.suggest_memory_format());
  scientific_call(337, {quantized_storage_view(self), quantized_storage_view(result)}, {2, groups}, {});
  return result;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("mean.dim", TORCH_FN(quantized_mean));
  m.impl("mean.out", TORCH_FN(quantized_mean_out));
  m.impl("std.correction", TORCH_FN(quantized_std));
  m.impl("std.correction_out", TORCH_FN(quantized_std_out));
  m.impl("channel_shuffle", TORCH_FN(quantized_channel_shuffle));
}
