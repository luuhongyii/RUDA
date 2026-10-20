template<int Dimensions>
std::array<int64_t, 3> quantized_pool_parameter(at::IntArrayRef value, int64_t fill, const char* name) {
  TORCH_CHECK(value.size() == 1 || value.size() == Dimensions,
              name, " must have one or ", Dimensions, " elements");
  std::array<int64_t, 3> result{fill, fill, fill};
  for (int i = 0; i < Dimensions; ++i) result[3 - Dimensions + i] = value[value.size() == 1 ? 0 : i];
  return result;
}

template<int Dimensions>
at::Tensor quantized_pool_view(const at::Tensor& input) {
  auto raw = input.is_quantized() ? quantized_storage_view(input) : input;
  if (raw.dim() == Dimensions + 1) raw = raw.unsqueeze(0);
  for (int i = Dimensions; i < 3; ++i) raw = raw.unsqueeze(2);
  return raw;
}

template<int Dimensions>
at::MemoryFormat quantized_pool_format(const at::Tensor& input) {
  if constexpr (Dimensions == 2) {
    if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) return input.suggest_memory_format();
  } else if constexpr (Dimensions == 3) {
    if (input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) return input.suggest_memory_format();
  }
  return at::MemoryFormat::Contiguous;
}

at::Tensor& quantized_pool_copy_result(const at::Tensor& value, at::Tensor& out) {
  TORCH_CHECK(out.is_quantized() && out.scalar_type() == value.scalar_type() && out.device() == value.device(),
              "quantized pooling output must match result dtype and device");
  TORCH_CHECK(out.qscheme() == value.qscheme(), "quantized pooling output qscheme mismatch");
  at::assert_no_internal_overlap(out);
  if (at::native::resize_output_check(out, value.sizes())) resize_tensor(out, value.sizes(), std::nullopt);
  quantized_copy_from(value, out, false);
  return out;
}

template<int Dimensions>
at::Tensor quantized_max_pool(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  validate(input.device());
  const bool plain_byte = !input.is_quantized() && input.scalar_type() == at::kByte && Dimensions == 2;
  if (!plain_byte) quantized_arithmetic_input(input);
  if constexpr (Dimensions == 1) {
    at::native::check_max_pool1d(input, kernel, stride, padding, dilation, ceil_mode);
    const auto step = stride.empty() ? kernel[0] : stride[0];
    return quantized_max_pool<2>(input.unsqueeze(-2), {1, kernel[0]}, {1, step},
                                {0, padding[0]}, {1, dilation[0]}, ceil_mode).squeeze(-2);
  } else if constexpr (Dimensions == 2) {
    TORCH_CHECK(stride.empty() || stride.size() == 2, "quantized max_pool2d expects no stride or two strides");
  }
  TORCH_CHECK(input.dim() == Dimensions + 1 || input.dim() == Dimensions + 2,
              "quantized max_pool input rank mismatch");
  if constexpr (Dimensions == 3) {
    TORCH_CHECK(input.dim() == 5, "quantized max_pool3d expects rank5 input");
    TORCH_CHECK(kernel.size() == 3 && padding.size() == 3 && dilation.size() == 3,
                "quantized max_pool3d expects three kernel, padding and dilation values");
    TORCH_CHECK(stride.empty() || stride.size() == 3, "quantized max_pool3d expects no stride or three strides");
  }
  auto k = quantized_pool_parameter<Dimensions>(kernel, 1, "kernel_size");
  auto s = stride.empty() ? k : quantized_pool_parameter<Dimensions>(stride, 1, "stride");
  auto p = quantized_pool_parameter<Dimensions>(padding, 0, "padding");
  auto d = quantized_pool_parameter<Dimensions>(dilation, 1, "dilation");
  auto shape = input.sizes().vec();
  TORCH_CHECK(input.size(-Dimensions - 1) > 0, "quantized max_pool channels must be nonempty");
  for (int i = 0; i < Dimensions; ++i) {
    const auto j = 3 - Dimensions + i;
    const auto axis = input.dim() - Dimensions + i;
    TORCH_CHECK(k[j] > 0 && s[j] > 0 && d[j] > 0, "kernel, stride and dilation must be positive");
    TORCH_CHECK(p[j] <= k[j] / 2, "padding must be at most half the kernel size");
    TORCH_CHECK(input.size(axis) > 0, "quantized max_pool spatial dimensions must be nonempty");
    shape[axis] = at::native::pooling_output_shape<int64_t>(input.size(axis), k[j], p[j], s[j], d[j], ceil_mode);
    TORCH_CHECK(shape[axis] > 0, "quantized max_pool output size is too small");
  }
  const auto format = quantized_pool_format<Dimensions>(input);
  auto out = plain_byte ? at::empty(shape, input.options().memory_format(format)) :
      quantized_empty_affine(shape, input.scalar_type(), input.layout(), input.device(), false,
                             input.q_scale(), input.q_zero_point(), format);
  scientific_call(562, {quantized_pool_view<Dimensions>(input), quantized_pool_view<Dimensions>(out)},
      {k[0], k[1], k[2], s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2]}, {});
  return out;
}

template<int Dimensions>
at::Tensor& quantized_max_pool_out(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor& out) {
  return quantized_pool_copy_result(quantized_max_pool<Dimensions>(input, kernel, stride, padding, dilation, ceil_mode), out);
}

template<int Dimensions>
at::Tensor quantized_max_pool_entry(const at::Tensor& input, std::vector<int64_t> kernel, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  return quantized_max_pool<Dimensions>(input, kernel, stride, padding, dilation, ceil_mode);
}

template<int Dimensions>
at::Tensor quantized_avg_pool(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor) {
  quantized_arithmetic_input(input);
  TORCH_CHECK(input.dim() == Dimensions + 1 || input.dim() == Dimensions + 2,
              "quantized avg_pool input rank mismatch");
  TORCH_CHECK(!divisor || *divisor != 0, "divisor must be not zero");
  auto k = quantized_pool_parameter<Dimensions>(kernel, 1, "kernel_size");
  auto s = stride.empty() ? k : quantized_pool_parameter<Dimensions>(stride, 1, "stride");
  auto p = quantized_pool_parameter<Dimensions>(padding, 0, "padding");
  auto shape = input.sizes().vec();
  for (int i = 0; i < 3; ++i) {
    k[i] = c10::checked_convert<int>(k[i], "int");
    s[i] = c10::checked_convert<int>(s[i], "int");
    p[i] = c10::checked_convert<int>(p[i], "int");
    TORCH_CHECK(k[i] > 0 && s[i] > 0, "kernel and stride must be positive");
  }
  for (int i = 0; i < Dimensions; ++i) {
    const auto j = 3 - Dimensions + i;
    const auto axis = input.dim() - Dimensions + i;
    TORCH_CHECK(input.size(axis) > 0, "quantized avg_pool spatial dimensions must be nonempty");
    shape[axis] = at::native::pooling_output_shape<int64_t>(input.size(axis), k[j], p[j], s[j], 1, ceil_mode);
    TORCH_CHECK(shape[axis] > 0, "quantized avg_pool output size is too small");
  }
  auto format = quantized_pool_format<Dimensions>(input);
  const bool channel_math = Dimensions == 3 || input.is_contiguous(at::MemoryFormat::ChannelsLast);
  if constexpr (Dimensions == 3) {
    TORCH_CHECK(input.dim() == 5, "quantized avg_pool3d expects rank5 input");
    format = at::MemoryFormat::ChannelsLast3d;
  }
  auto out = quantized_empty_affine(shape, input.scalar_type(), input.layout(), input.device(), false,
                                   input.q_scale(), input.q_zero_point(), format);
  scientific_call(563, {quantized_pool_view<Dimensions>(input), quantized_pool_view<Dimensions>(out)},
      {k[0], k[1], k[2], s[0], s[1], s[2], p[0], p[1], p[2], int64_t(count_include_pad),
       divisor.value_or(0), input.q_zero_point(), int64_t(channel_math)}, {input.q_scale()});
  return out;
}

template<int Dimensions>
at::Tensor& quantized_avg_pool_out(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor, at::Tensor& out) {
  return quantized_pool_copy_result(quantized_avg_pool<Dimensions>(input, kernel, stride, padding, ceil_mode, count_include_pad, divisor), out);
}

at::Tensor quantized_avg_pool1d(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "avg_pool1d expects rank2 or rank3 input");
  TORCH_CHECK(kernel.size() == 1 && padding.size() == 1 && (stride.empty() || stride.size() == 1),
              "avg_pool1d expects one kernel, padding and stride value");
  const auto step = stride.empty() ? kernel[0] : stride[0];
  return quantized_avg_pool<2>(input.unsqueeze(-2), {1, kernel[0]}, {1, step}, {0, padding[0]},
                               ceil_mode, count_include_pad, std::nullopt).squeeze(-2);
}

at::Tensor& quantized_avg_pool1d_out(const at::Tensor& input, at::IntArrayRef kernel, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, at::Tensor& out) {
  return quantized_pool_copy_result(quantized_avg_pool1d(input, kernel, stride, padding, ceil_mode, count_include_pad), out);
}

template<int Dimensions>
at::Tensor quantized_adaptive_avg_pool(const at::Tensor& input, at::IntArrayRef output_size) {
  quantized_arithmetic_input(input);
  TORCH_CHECK(input.dim() == Dimensions + 1 || input.dim() == Dimensions + 2,
              "quantized adaptive_avg_pool input rank mismatch");
  TORCH_CHECK(output_size.size() == Dimensions, "adaptive pooling output_size length mismatch");
  if constexpr (Dimensions == 1) {
    return quantized_adaptive_avg_pool<2>(input.unsqueeze(-2), {1, output_size[0]}).squeeze(-2);
  }
  for (int64_t i = 1; i < input.dim(); ++i) {
    TORCH_CHECK(input.size(i) > 0, "quantized adaptive_avg_pool requires nonempty non-batch dimensions");
  }
  auto shape = input.sizes().vec();
  for (int i = 0; i < Dimensions; ++i) {
    TORCH_CHECK(output_size[i] >= 0, "adaptive pooling output_size must be nonnegative");
    shape[input.dim() - Dimensions + i] = output_size[i];
  }
  const bool channel_math = input.is_contiguous(at::MemoryFormat::ChannelsLast) ||
                            input.is_contiguous(at::MemoryFormat::ChannelsLast3d);
  const auto format = channel_math ? input.suggest_memory_format() : at::MemoryFormat::Contiguous;
  auto out = quantized_empty_affine(shape, input.scalar_type(), input.layout(), input.device(), false,
                                   input.q_scale(), input.q_zero_point(), format);
  scientific_call(564, {quantized_pool_view<Dimensions>(input), quantized_pool_view<Dimensions>(out)},
      {input.q_zero_point(), int64_t(channel_math)}, {input.q_scale()});
  return out;
}

template<int Dimensions>
at::Tensor& quantized_adaptive_avg_pool_out(const at::Tensor& input, at::IntArrayRef output_size, at::Tensor& out) {
  return quantized_pool_copy_result(quantized_adaptive_avg_pool<Dimensions>(input, output_size), out);
}

at::Tensor& quantized_adaptive_avg_pool3d_out(const at::Tensor& input, at::IntArrayRef output_size, at::Tensor& out) {
  out = quantized_adaptive_avg_pool<3>(input, output_size);
  return out;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("max_pool1d", TORCH_FN(quantized_max_pool<1>));
  m.impl("max_pool2d", TORCH_FN(quantized_max_pool<2>));
  m.impl("max_pool3d", TORCH_FN(quantized_max_pool<3>));
  m.impl("quantized_max_pool1d", TORCH_FN(quantized_max_pool<1>));
  m.impl("quantized_max_pool2d", TORCH_FN(quantized_max_pool<2>));
  m.impl("quantized_max_pool3d", TORCH_FN(quantized_max_pool<3>));
  m.impl("quantized_max_pool1d.out", TORCH_FN(quantized_max_pool_out<1>));
  m.impl("quantized_max_pool2d.out", TORCH_FN(quantized_max_pool_out<2>));
  m.impl("quantized_max_pool3d.out", TORCH_FN(quantized_max_pool_out<3>));
  m.impl("avg_pool1d", TORCH_FN(quantized_avg_pool1d));
  m.impl("avg_pool1d.out", TORCH_FN(quantized_avg_pool1d_out));
  m.impl("avg_pool2d", TORCH_FN(quantized_avg_pool<2>));
  m.impl("avg_pool2d.out", TORCH_FN(quantized_avg_pool_out<2>));
  m.impl("avg_pool3d", TORCH_FN(quantized_avg_pool<3>));
  m.impl("avg_pool3d.out", TORCH_FN(quantized_avg_pool_out<3>));
  m.impl("adaptive_avg_pool1d", TORCH_FN(quantized_adaptive_avg_pool<1>));
  m.impl("adaptive_avg_pool1d.out", TORCH_FN(quantized_adaptive_avg_pool_out<1>));
  m.impl("adaptive_avg_pool2d", TORCH_FN(quantized_adaptive_avg_pool<2>));
  m.impl("adaptive_avg_pool2d.out", TORCH_FN(quantized_adaptive_avg_pool_out<2>));
  m.impl("_adaptive_avg_pool2d", TORCH_FN(quantized_adaptive_avg_pool<2>));
  m.impl("_adaptive_avg_pool2d.out", TORCH_FN(quantized_adaptive_avg_pool_out<2>));
  m.impl("adaptive_avg_pool3d", TORCH_FN(quantized_adaptive_avg_pool<3>));
  m.impl("adaptive_avg_pool3d.out", TORCH_FN(quantized_adaptive_avg_pool3d_out));
  m.impl("_adaptive_avg_pool3d", TORCH_FN(quantized_adaptive_avg_pool<3>));
  m.impl("_adaptive_avg_pool3d.out", TORCH_FN(quantized_adaptive_avg_pool_out<3>));
}
TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("max_pool1d", TORCH_FN(quantized_max_pool_entry<1>));
  m.impl("max_pool2d", TORCH_FN(quantized_max_pool_entry<2>));
}
TORCH_LIBRARY_IMPL(quantized, PrivateUse1, m) {
  m.impl("max_pool2d", TORCH_FN(quantized_max_pool_entry<2>));
}
