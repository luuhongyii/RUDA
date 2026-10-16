at::Tensor quantized_cat_compute(const at::ITensorListRef& tensors, int64_t dim,
    double scale, int64_t zero, bool relu, bool warn_qparams) {
  auto inputs = tensors.materialize();
  TORCH_CHECK(!inputs.empty(), "torch.cat(): expected a non-empty list of Tensors");
  at::native::check_cat_no_zero_dim(inputs);
  dim = at::legacy_cat_wrap_dim(dim, inputs);
  const auto& first = inputs[0].get();
  quantized_arithmetic_input(first);
  bool same_qparams = true;
  auto format = first.suggest_memory_format();
  size_t valid = inputs.size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i].get();
    quantized_arithmetic_pair(first, input);
    if (input.suggest_memory_format() != format) format = at::MemoryFormat::Contiguous;
    if (i != 0)
      same_qparams &= std::abs(input.q_scale() - first.q_scale()) < 1e-4 && input.q_zero_point() == first.q_zero_point();
    if (valid == inputs.size() && !at::native::cat_should_skip_tensor(input)) valid = i;
  }
  if (warn_qparams && !same_qparams)
    TORCH_WARN("All inputs of this cat operator must share the same quantization parameters. Otherwise large numerical inaccuracies may occur.");
  at::DimVector shape{0};
  if (valid != inputs.size()) {
    shape = at::DimVector(inputs[valid].get().sizes());
    int64_t length = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i].get();
      if (at::native::cat_should_skip_tensor(input)) continue;
      at::native::check_cat_shape_except_dim(inputs[valid].get(), input, dim, i);
      TORCH_CHECK(input.size(dim) <= std::numeric_limits<int64_t>::max() - length, "quantized cat dimension overflow");
      length += input.size(dim);
    }
    shape[dim] = length;
  }
  auto out = quantized_empty_affine(shape, first.scalar_type(), first.layout(), first.device(), false, scale, zero, format);
  auto raw_out = quantized_storage_view(out);
  int64_t offset = 0;
  for (const at::Tensor& input : inputs) {
    if (at::native::cat_should_skip_tensor(input)) continue;
    if (input.numel() != 0) {
      auto target = raw_out.narrow(dim, offset, input.size(dim));
      scientific_call(556, {quantized_storage_view(input), target},
          {input.q_zero_point(), zero, int64_t(relu)}, {input.q_scale(), scale});
    }
    offset += input.size(dim);
  }
  return out;
}

at::Tensor quantized_cat_aten(const at::ITensorListRef& tensors, int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "torch.cat(): expected a non-empty list of Tensors");
  const at::Tensor& first = tensors.front();
  return quantized_cat_compute(tensors, dim, first.q_scale(), first.q_zero_point(), false, true);
}
at::Tensor& quantized_cat_aten_out(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out) {
  quantized_arithmetic_input(out);
  auto result = quantized_cat_compute(tensors, dim, out.q_scale(), out.q_zero_point(), false, false);
  quantized_copy_from(result, out, false);
  return out;
}
template<bool Relu>
at::Tensor quantized_cat(const c10::List<at::Tensor>& tensors, int64_t dim,
                         std::optional<double> scale, std::optional<int64_t> zero) {
  TORCH_CHECK(!tensors.empty(), "quantized::cat(): expected a non-empty list of Tensors");
  auto first = tensors.get(0);
  return quantized_cat_compute(at::ITensorListRef(tensors), dim,
      scale.value_or(first.q_scale()), zero.value_or(first.q_zero_point()), Relu, false);
}
template<bool Relu>
at::Tensor quantized_cat_out(const c10::List<at::Tensor>& tensors, int64_t dim, at::Tensor out) {
  quantized_arithmetic_input(out);
  auto result = quantized_cat_compute(at::ITensorListRef(tensors), dim, out.q_scale(), out.q_zero_point(), Relu, false);
  quantized_copy_from(result, out, false);
  return out;
}

at::Tensor quantized_prelu_qparams(at::Tensor self, const at::Tensor& weight, double scale, int64_t zero) {
  validate(self.device());
  quantized_storage_type(self.scalar_type());
  if (self.dim() < 1 || self.dim() > 5) {
    auto real = at::native::dequantize_quantized(self);
    return quantize_tensor_ruda(at::prelu(real, weight), scale, zero, at::kQUInt8);
  }
  quantized_arithmetic_pair(self, weight);
  auto raw_weight = quantized_storage_view(weight);
  if (weight.dim() != self.dim()) {
    at::DimVector shape(self.dim(), 1);
    if (self.dim() > 1) shape[1] = weight.numel();
    raw_weight = raw_weight.reshape(shape);
  }
  auto shape = at::infer_size_dimvector(self.sizes(), raw_weight.sizes());
  auto out = quantized_empty_affine(shape, self.scalar_type(), self.layout(), self.device(), false,
                                   scale, zero, self.suggest_memory_format());
  int64_t quantized_one = 0;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "ruda_quantized_prelu", [&] {
    quantized_one = at::native::quantize_val<scalar_t>(
        static_cast<float>(weight.q_scale()), weight.q_zero_point(), 1.0f).val_;
  });
  scientific_call(557, {quantized_storage_view(self).expand(shape), raw_weight.expand(shape), quantized_storage_view(out)},
      {self.q_zero_point(), weight.q_zero_point(), zero, quantized_one}, {self.q_scale(), weight.q_scale(), scale});
  return out;
}
at::Tensor quantized_prelu_kernel(const at::Tensor& self, const at::Tensor& weight) {
  return quantized_prelu_qparams(self, weight, self.q_scale(), self.q_zero_point());
}

at::Tensor quantized_softmax(const at::Tensor& self, int64_t dim, double scale, int64_t zero) {
  validate(self.device());
  quantized_storage_type(self.scalar_type());
  dim = at::maybe_wrap_dim(dim, self.dim());
  if (self.qscheme() != c10::kPerTensorAffine) {
    auto real = at::native::dequantize_quantized(self);
    return quantize_tensor_ruda(at::softmax(real, dim, std::nullopt), scale, zero, self.scalar_type());
  }
  auto out = quantized_empty_affine(self.sizes(), self.scalar_type(), self.layout(), self.device(), false,
                                   scale, zero, at::MemoryFormat::Contiguous);
  auto input = quantized_storage_view(self); auto output = quantized_storage_view(out);
  if (self.dim() == 0) { input = input.reshape({1}); output = output.reshape({1}); }
  scientific_call(558, {input, output}, {self.q_zero_point(), zero, dim}, {self.q_scale(), scale});
  return out;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("cat", TORCH_FN(quantized_cat_aten));
  m.impl("cat.out", TORCH_FN(quantized_cat_aten_out));
  m.impl("_prelu_kernel", TORCH_FN(quantized_prelu_kernel));
}
TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("cat", TORCH_FN(quantized_cat<false>));
  m.impl("cat_relu", TORCH_FN(quantized_cat<true>));
  m.impl("cat_out", TORCH_FN(quantized_cat_out<false>));
  m.impl("cat_relu_out", TORCH_FN(quantized_cat_out<true>));
  m.impl("prelu", TORCH_FN(quantized_prelu_qparams));
  m.impl("softmax", TORCH_FN(quantized_softmax));
}
