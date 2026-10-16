c10::ScalarType quantized_storage_type(c10::ScalarType dtype) {
  switch (dtype) {
    case at::kQInt8: return at::kChar;
    case at::kQUInt8: return at::kByte;
    case at::kQInt32: return at::kInt;
    default: TORCH_CHECK(false, "RUDA quantized tensors require qint8, quint8 or qint32 storage");
  }
}

at::Tensor quantized_storage_view(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_quantized(), "expected a quantized tensor");
  const auto dtype = quantized_storage_type(tensor.scalar_type());
  auto result = at::detail::make_tensor<c10::TensorImpl>(
      at::Storage(tensor.storage()),
      c10::DispatchKeySet(tensor.options().dtype(dtype).computeDispatchKey()),
      c10::scalarTypeToTypeMeta(dtype));
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(tensor.sizes(), tensor.strides());
  impl->set_storage_offset(tensor.storage_offset());
  return result;
}

void quantized_device_options(std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin) {
  TORCH_CHECK(dtype.has_value(), "quantized tensor creation requires an explicit dtype");
  quantized_storage_type(*dtype);
  validate(c10::device_or_default(device));
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided && !c10::pinned_memory_or_default(pin));
  TORCH_CHECK(at::globalContext().qEngine() != at::QEngine::QNNPACK,
              "QNNPACK's CPU quantized allocator cannot create a RUDA GPU tensor");
}

at::Tensor quantized_empty(at::IntArrayRef size, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin, std::optional<c10::MemoryFormat> format) {
  quantized_device_options(dtype, layout, device, pin);
  return at::native::empty_unknown_quantized(size, dtype, layout, device, pin, format);
}

at::Tensor quantized_empty_affine(at::IntArrayRef size, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin, double scale, int64_t zero, std::optional<c10::MemoryFormat> format) {
  quantized_device_options(dtype, layout, device, pin);
  return at::native::empty_affine_quantized(size, dtype, layout, device, pin, scale, zero, format);
}

at::Tensor quantized_empty_channel(at::IntArrayRef size, const at::Tensor& scales,
    const at::Tensor& zeros, int64_t axis, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout, std::optional<c10::Device> device,
    std::optional<bool> pin, std::optional<c10::MemoryFormat> format) {
  quantized_device_options(dtype, layout, device, pin);
  return at::native::empty_per_channel_affine_quantized(size, scales, zeros, axis, dtype, layout, device, pin, format);
}

void quantized_affine_scalar(const at::Tensor& real, const at::Tensor& quantized,
                            double scale, int64_t zero, bool reverse) {
  validate(real.device()); validate(quantized.device());
  auto raw = quantized_storage_view(quantized);
  auto value = reverse ? real : scientific_resolve(real);
  scientific_call(549, {value, raw}, {zero, int64_t(reverse)}, {scale});
}

void quantized_affine_channel(const at::Tensor& real, const at::Tensor& quantized,
    const at::Tensor& scales, const at::Tensor& zeros, int64_t axis, bool reverse, bool floating) {
  validate(real.device()); validate(quantized.device());
  TORCH_CHECK(scales.device() == real.device() && zeros.device() == real.device(),
              "per-channel quantization parameters must be on the input device");
  auto raw = quantized_storage_view(quantized);
  int64_t minimum = 0, maximum = 255;
  if (raw.scalar_type() == at::kChar) { minimum = -128; maximum = 127; }
  else if (raw.scalar_type() == at::kInt) {
    minimum = std::numeric_limits<int32_t>::min(); maximum = std::numeric_limits<int32_t>::max();
  }
  if (!floating) {
    TORCH_CHECK(!zeros.lt(minimum).any().item<bool>(), "quantized zero_point is below lower bound");
    TORCH_CHECK(!zeros.gt(maximum).any().item<bool>(), "quantized zero_point is above upper bound");
  }
  auto value = reverse ? real : scientific_resolve(real);
  auto scale_values = scientific_resolve(scales).to(floating ? at::kFloat : at::kDouble);
  auto zero_values = scientific_resolve(zeros).to(floating ? at::kFloat : at::kLong);
  scientific_call(floating ? 551 : 550, {value, raw, scale_values, zero_values},
                  {axis, int64_t(reverse)}, {});
}

void quantize_scalar_ruda(const at::Tensor& real, at::Tensor& quantized, double scale, int64_t zero) {
  quantized_affine_scalar(real, quantized, scale, zero, false);
}
void dequantize_scalar_ruda(const at::Tensor& quantized, at::Tensor& real, double scale, int64_t zero) {
  quantized_affine_scalar(real, quantized, scale, zero, true);
}
void quantize_channel_ruda(const at::Tensor& real, at::Tensor& quantized,
    const at::Tensor& scales, const at::Tensor& zeros, int64_t axis) {
  quantized_affine_channel(real, quantized, scales, zeros, axis, false, false);
}
void dequantize_channel_ruda(const at::Tensor& quantized, at::Tensor& real,
    const at::Tensor& scales, const at::Tensor& zeros, int64_t axis) {
  quantized_affine_channel(real, quantized, scales, zeros, axis, true, false);
}
void quantize_float_channel_ruda(const at::Tensor& real, at::Tensor& quantized,
    const at::Tensor& scales, const at::Tensor& zeros, int64_t axis) {
  quantized_affine_channel(real, quantized, scales, zeros, axis, false, true);
}
void dequantize_float_channel_ruda(const at::Tensor& quantized, at::Tensor& real,
    const at::Tensor& scales, const at::Tensor& zeros, int64_t axis) {
  quantized_affine_channel(real, quantized, scales, zeros, axis, true, true);
}

at::Tensor quantize_tensor_ruda(const at::Tensor& self, double scale, int64_t zero, c10::ScalarType dtype) {
  quantized_device_options(dtype, self.layout(), self.device(), false);
  return at::native::quantize_per_tensor(self, scale, zero, dtype);
}
at::Tensor quantize_tensor_qparams_ruda(const at::Tensor& self, const at::Tensor& scale,
    const at::Tensor& zero, c10::ScalarType dtype) {
  quantized_device_options(dtype, self.layout(), self.device(), false);
  return at::native::quantize_per_tensor_tensor_qparams(self, scale, zero, dtype);
}
at::Tensor quantize_channel_tensor_ruda(const at::Tensor& self, const at::Tensor& scales,
    const at::Tensor& zeros, int64_t axis, c10::ScalarType dtype) {
  quantized_device_options(dtype, self.layout(), self.device(), false);
  return at::native::quantize_per_channel(self, scales, zeros, axis, dtype);
}

at::Tensor quantized_int_repr(const at::Tensor& self) {
  validate(self.device());
  auto raw = quantized_storage_view(self);
  auto result = at::empty(self.sizes(), raw.options(), self.suggest_memory_format());
  copy_from(raw, result, false);
  return result;
}

at::Tensor quantized_make_tensor(const at::Tensor& self, double scale, int64_t zero) {
  validate(self.device());
  auto result = quantized_empty_affine(self.sizes(), at::toQIntType(self.scalar_type()),
      self.layout(), self.device(), false, scale, zero, at::MemoryFormat::Contiguous);
  copy_from(self, quantized_storage_view(result), false);
  return result;
}
at::Tensor quantized_make_channel(const at::Tensor& self, const at::Tensor& scales,
    const at::Tensor& zeros, int64_t axis) {
  validate(self.device());
  auto result = quantized_empty_channel(self.sizes(), scales, zeros, axis,
      at::toQIntType(self.scalar_type()), self.layout(), self.device(), false, at::MemoryFormat::Contiguous);
  copy_from(self, quantized_storage_view(result), false);
  return result;
}

at::Tensor quantized_copy_from(const at::Tensor& source, const at::Tensor& dest, bool non_blocking) {
  TORCH_CHECK(dest.is_quantized(), "Copying a quantized tensor to a non-quantized tensor requires dequantize()");
  auto& output = const_cast<at::Tensor&>(dest);
  if (!source.is_quantized()) {
    TORCH_CHECK(source.scalar_type() == at::kFloat, "quantized copy requires a Float source tensor");
    TORCH_CHECK((source.is_contiguous() && dest.is_contiguous()) ||
                (source.is_contiguous(at::MemoryFormat::ChannelsLast) && dest.is_contiguous(at::MemoryFormat::ChannelsLast)),
                "quantized copy from Float requires contiguous or channels-last tensors");
    TORCH_CHECK(source.sizes() == dest.sizes(), "quantized copy from Float requires matching shapes");
    const auto scheme = dest.qscheme();
    if (scheme == at::kPerChannelAffine || scheme == at::kPerChannelAffineFloatQParams || scheme == at::kPerChannelSymmetric) {
      at::native::quantize_tensor_per_channel_affine(source, output,
          dest.q_per_channel_scales(), dest.q_per_channel_zero_points(), dest.q_per_channel_axis());
    } else {
      at::native::quantize_tensor_per_tensor_affine(source, output, dest.q_scale(), dest.q_zero_point());
    }
    return dest;
  }
  TORCH_CHECK(source.scalar_type() == dest.scalar_type(), "quantized copy requires matching storage dtypes");
  TORCH_CHECK(source.qscheme() == dest.qscheme(), "quantized copy requires matching qschemes");
  at::assert_no_internal_overlap(dest);
  if (source.is_same(dest)) return dest;
  auto quantizer = source.quantizer();
  if (source.device() != dest.device() && (source.qscheme() == at::kPerChannelAffine ||
                                          source.qscheme() == at::kPerChannelAffineFloatQParams)) {
    quantizer = at::make_per_channel_affine_quantizer(
        source.q_per_channel_scales().to(dest.device()), source.q_per_channel_zero_points().to(dest.device()),
        source.q_per_channel_axis(), source.scalar_type());
  }
  copy_from(quantized_storage_view(source), quantized_storage_view(dest), non_blocking);
  at::set_quantizer_(dest, quantizer);
  return dest;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("quantize_per_tensor", quantize_tensor_ruda);
  m.impl("quantize_per_tensor.tensor_qparams", quantize_tensor_qparams_ruda);
  m.impl("quantize_per_channel", quantize_channel_tensor_ruda);
  m.impl("quantize_per_tensor.out", TORCH_FN(at::native::quantize_per_tensor_out));
  m.impl("quantize_per_tensor.tensor_qparams_out", TORCH_FN(at::native::quantize_per_tensor_tensor_qparams_out));
  m.impl("quantize_per_channel.out", TORCH_FN(at::native::quantize_per_channel_out));
  m.impl("dequantize.self", TORCH_FN(at::native::dequantize_cpu_or_cuda));
  m.impl("dequantize.self_out", TORCH_FN(at::native::dequantize_self_out));
  m.impl("_make_per_tensor_quantized_tensor", quantized_make_tensor);
  m.impl("_make_per_channel_quantized_tensor", quantized_make_channel);
  m.impl("_make_per_tensor_quantized_tensor.out", TORCH_FN(at::native::_make_per_tensor_quantized_tensor_out));
  m.impl("_make_per_channel_quantized_tensor.out", TORCH_FN(at::native::_make_per_channel_quantized_tensor_out));
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("empty.memory_format", quantized_empty);
  m.impl("empty_like", TORCH_FN(at::native::empty_like_quantized));
  m.impl("empty_quantized", TORCH_FN(at::native::empty_quantized));
  m.impl("empty_quantized.out", TORCH_FN(at::native::empty_quantized_out));
  m.impl("_empty_affine_quantized", quantized_empty_affine);
  m.impl("_empty_affine_quantized.out", TORCH_FN(at::native::_empty_affine_quantized_out_symint));
  m.impl("_empty_per_channel_affine_quantized", quantized_empty_channel);
  m.impl("_empty_per_channel_affine_quantized.out", TORCH_FN(at::native::_empty_per_channel_affine_quantized_out_symint));
  m.impl("quantize_per_tensor.out", TORCH_FN(at::native::quantize_per_tensor_out));
  m.impl("quantize_per_tensor.tensor_qparams_out", TORCH_FN(at::native::quantize_per_tensor_tensor_qparams_out));
  m.impl("quantize_per_channel.out", TORCH_FN(at::native::quantize_per_channel_out));
  m.impl("_make_per_tensor_quantized_tensor.out", TORCH_FN(at::native::_make_per_tensor_quantized_tensor_out));
  m.impl("_make_per_channel_quantized_tensor.out", TORCH_FN(at::native::_make_per_channel_quantized_tensor_out));
  m.impl("dequantize.self", TORCH_FN(at::native::dequantize_quantized));
  m.impl("dequantize.self_out", TORCH_FN(at::native::dequantize_self_out));
  m.impl("int_repr", quantized_int_repr);
  m.impl("int_repr.out", TORCH_FN(at::native::int_repr_out));
  m.impl("q_scale", TORCH_FN(at::native::q_scale_quant));
  m.impl("q_zero_point", TORCH_FN(at::native::q_zero_point_quant));
  m.impl("q_per_channel_scales", TORCH_FN(at::native::q_per_channel_scales));
  m.impl("q_per_channel_scales.out", TORCH_FN(at::native::q_per_channel_scales_out));
  m.impl("q_per_channel_zero_points", TORCH_FN(at::native::q_per_channel_zero_points));
  m.impl("q_per_channel_zero_points.out", TORCH_FN(at::native::q_per_channel_zero_points_out));
  m.impl("q_per_channel_axis", TORCH_FN(at::native::q_per_channel_axis));
  m.impl("qscheme", TORCH_FN(at::native::qscheme_quant));
  m.impl("clone", TORCH_FN(at::native::quantized_clone));
  m.impl("copy_", copy_);
  m.impl("_copy_from", quantized_copy_from);
  m.impl("as_strided", TORCH_FN(at::native::as_strided_qtensorimpl));
  m.impl("as_strided_", TORCH_FN(at::native::as_strided__symint));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("squeeze", [](const at::Tensor& self) { return at::native::squeeze_quantized(self); });
  m.impl("squeeze.dim", [](const at::Tensor& self, int64_t dim) { return at::native::squeeze_quantized(self, dim); });
  m.impl("squeeze.dims", [](const at::Tensor& self, at::IntArrayRef dims) { return at::native::squeeze_quantized(self, dims); });
  m.impl("unsqueeze", TORCH_FN(at::native::unsqueeze_quantized));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(at::native::set_storage_quantized_));
}
