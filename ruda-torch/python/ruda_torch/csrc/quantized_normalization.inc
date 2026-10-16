at::Tensor quantized_norm_parameter(const at::Tensor& input, const at::Tensor& parameter, int64_t count) {
  if (!parameter.defined()) return at::empty({0}, input.options().dtype(at::kFloat));
  TORCH_CHECK(parameter.device() == input.device() && parameter.scalar_type() == at::kFloat,
              "quantized normalization parameters must be Float tensors on the input device");
  TORCH_CHECK(parameter.numel() == count, "quantized normalization parameter size mismatch");
  return scientific_resolve(parameter).reshape({count});
}

template<int Kind>
at::Tensor quantized_batch_norm_entry(at::Tensor input, std::optional<at::Tensor> weight,
    std::optional<at::Tensor> bias, at::Tensor mean, at::Tensor variance,
    double eps, double scale, int64_t zero) {
  validate(input.device()); quantized_storage_type(input.scalar_type());
  const auto rank = input.dim();
  if constexpr (Kind < 2) {
    TORCH_CHECK(rank >= 2 && rank <= 5, "quantized::batch_norm only supports rank2 through rank5 inputs");
  }
  TORCH_CHECK(weight.has_value(), "Weight must be provided");
  TORCH_CHECK(bias.has_value(), "Bias must be provided");
  if (input.numel() == 0) return input.clone();
  if constexpr (Kind / 2 == 1) {
    TORCH_CHECK(rank == 2 || rank == 3, "batch_norm1d requires rank2 or rank3 input");
  } else if constexpr (Kind / 2 == 2) {
    TORCH_CHECK(rank == 4, "batch_norm2d requires rank4 input");
  } else if constexpr (Kind / 2 == 3) {
    TORCH_CHECK(rank == 5, "batch_norm3d requires rank5 input");
  }
  quantized_arithmetic_input(input);
  const auto channels = input.size(1);
  auto w = quantized_norm_parameter(input, *weight, channels);
  auto b = quantized_norm_parameter(input, *bias, channels);
  auto m = quantized_norm_parameter(input, mean, channels);
  auto v = quantized_norm_parameter(input, variance, channels);
  TORCH_CHECK(w.numel() == channels && b.numel() == channels && m.numel() == channels && v.numel() == channels,
              "BatchNorm weight, bias, mean and variance must be defined");
  auto alpha = at::empty({channels}, input.options().dtype(at::kFloat));
  auto beta = at::empty_like(alpha);
  const auto format = rank == 4 ? at::MemoryFormat::ChannelsLast :
                      rank == 5 ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::Contiguous;
  auto out = quantized_empty_affine(input.sizes(), input.scalar_type(), input.layout(), input.device(), false,
                                   scale, zero, format);
  const auto spatial = c10::multiply_integers(input.sizes().begin() + 2, input.sizes().end());
  scientific_call(559, {w, b, m, v, alpha, beta}, {}, {eps, input.q_scale(), scale});
  scientific_call(560, {quantized_storage_view(input), alpha, beta, quantized_storage_view(out)},
      {input.q_zero_point(), zero, channels, spatial, int64_t(Kind % 2)}, {});
  return out;
}

at::Tensor quantized_batch_norm_aten(const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, const at::Tensor& mean, const at::Tensor& variance,
    double eps, double scale, int64_t zero) {
  return quantized_batch_norm_entry<0>(input, weight, bias, mean, variance, eps, scale, zero);
}
at::Tensor& quantized_batch_norm_out(const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, const at::Tensor& mean, const at::Tensor& variance,
    double eps, double scale, int64_t zero, at::Tensor& out) {
  auto value = quantized_batch_norm_entry<0>(input, weight, bias, mean, variance, eps, scale, zero);
  TORCH_CHECK(out.is_quantized() && out.scalar_type() == value.scalar_type() && out.device() == value.device(),
              "quantized BatchNorm output must match the result dtype and device");
  TORCH_CHECK(out.qscheme() == value.qscheme(), "quantized BatchNorm output quantization scheme mismatch");
  at::assert_no_internal_overlap(out);
  if (at::native::resize_output_check(out, value.sizes())) resize_tensor(out, value.sizes(), std::nullopt);
  quantized_copy_from(value, out, false);
  return out;
}

void quantized_normalize_call(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& output, int64_t rows, int64_t width, int64_t channels, int64_t spatial,
    bool per_channel, bool channels_last, double eps) {
  quantized_arithmetic_input(input);
  const auto affine = per_channel ? channels : width;
  auto w = quantized_norm_parameter(input, weight, affine);
  auto b = quantized_norm_parameter(input, bias, affine);
  scientific_call(561, {quantized_storage_view(input), w, b, quantized_storage_view(output)},
      {int64_t(per_channel), rows, width, channels, spatial, input.q_zero_point(), output.q_zero_point(), int64_t(channels_last)},
      {eps, input.q_scale(), output.q_scale()});
}

at::Tensor quantized_layer_norm(at::Tensor input, std::vector<int64_t> shape,
    std::optional<at::Tensor> weight, std::optional<at::Tensor> bias, double eps, double scale, int64_t zero) {
  validate(input.device()); quantized_storage_type(input.scalar_type());
  auto w = weight.value_or(at::Tensor()); auto b = bias.value_or(at::Tensor());
  const at::IntArrayRef normalized_shape(shape);
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ", normalized_shape);
  TORCH_CHECK(!w.defined() || w.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ", w.sizes(), " and normalized_shape = ", normalized_shape);
  TORCH_CHECK(!b.defined() || b.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ", b.sizes(), " and normalized_shape = ", normalized_shape);
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
    std::stringstream message;
    message << "Given normalized_shape=" << normalized_shape << ", expected input with shape [*";
    for (auto size : normalized_shape) message << ", " << size;
    message << "], but got input of size" << input_shape;
    TORCH_CHECK(false, message.str());
  }
  const int axis = input_ndim - normalized_ndim;
  const int64_t rows = c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t width = c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());
  auto out = quantized_empty_affine(input.sizes(), input.scalar_type(), input.layout(), input.device(), false,
                                   scale, zero, at::MemoryFormat::Contiguous);
  if (rows > 0) quantized_normalize_call(input, w, b, out, rows, width, 1, 1, false, false, eps);
  return out;
}

at::Tensor quantized_group_norm(at::Tensor input, int64_t groups,
    std::optional<at::Tensor> weight, std::optional<at::Tensor> bias, double eps, double scale, int64_t zero) {
  validate(input.device()); quantized_storage_type(input.scalar_type());
  TORCH_CHECK(input.dim() >= 3, "quantized GroupNorm requires input rank at least3");
  TORCH_CHECK(groups > 0, "Expected num_groups to be positive");
  const auto channels = input.size(1);
  TORCH_CHECK(channels % groups == 0, "Expected channels to be divisible by groups");
  const auto spatial = c10::multiply_integers(input.sizes().begin() + 2, input.sizes().end());
  const int64_t rows = input.size(0) * groups;
  const int64_t width = (channels / groups) * spatial;
  const bool channels_last = input.is_contiguous(at::MemoryFormat::ChannelsLast);
  auto out = quantized_empty_affine(input.sizes(), input.scalar_type(), input.layout(), input.device(), false,
      scale, zero, channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous);
  if (rows > 0) quantized_normalize_call(input, weight.value_or(at::Tensor()), bias.value_or(at::Tensor()),
      out, rows, width, channels, spatial, true, channels_last, eps);
  return out;
}

at::Tensor quantized_instance_norm(at::Tensor input, std::optional<at::Tensor> weight,
    std::optional<at::Tensor> bias, double eps, double scale, int64_t zero) {
  TORCH_CHECK(input.dim() >= 3, "quantized InstanceNorm requires input rank at least3");
  TORCH_CHECK(input.size(1) > 0, "Expected 2nd dimension to be positive");
  return quantized_group_norm(input, input.size(1), weight, bias, eps, scale, zero);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("quantized_batch_norm", TORCH_FN(quantized_batch_norm_aten));
  m.impl("quantized_batch_norm.out", TORCH_FN(quantized_batch_norm_out));
}
TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("batch_norm", TORCH_FN(quantized_batch_norm_entry<0>));
  m.impl("batch_norm_relu", TORCH_FN(quantized_batch_norm_entry<1>));
  m.impl("batch_norm1d", TORCH_FN(quantized_batch_norm_entry<2>));
  m.impl("batch_norm1d_relu", TORCH_FN(quantized_batch_norm_entry<3>));
  m.impl("batch_norm2d", TORCH_FN(quantized_batch_norm_entry<4>));
  m.impl("batch_norm2d_relu", TORCH_FN(quantized_batch_norm_entry<5>));
  m.impl("batch_norm3d", TORCH_FN(quantized_batch_norm_entry<6>));
  m.impl("batch_norm3d_relu", TORCH_FN(quantized_batch_norm_entry<7>));
  m.impl("layer_norm", TORCH_FN(quantized_layer_norm));
  m.impl("group_norm", TORCH_FN(quantized_group_norm));
  m.impl("instance_norm", TORCH_FN(quantized_instance_norm));
}
