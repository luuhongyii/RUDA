template<int Dimensions>
at::Tensor quantized_resize_output(const at::Tensor& input, at::IntArrayRef output_size) {
  quantized_arithmetic_input(input);
  std::vector<int64_t> shape;
  if constexpr (Dimensions == 2) {
    const auto full_size = at::native::upsample_2d_common_check(input.sizes(), output_size);
    TORCH_CHECK(input.size(1) > 0, "quantized upsampling requires nonempty channels");
    shape.assign(full_size.begin(), full_size.end());
  } else {
    TORCH_CHECK(output_size.size() == 3, "quantized nearest3d requires three output sizes");
    TORCH_CHECK(input.dim() == 5 && input.numel() != 0, "quantized nearest3d requires a nonempty5D input");
    TORCH_CHECK(output_size[2] > 0, "quantized nearest3d output width must be positive");
    shape = {input.size(0), input.size(1), output_size[0], output_size[1], output_size[2]};
  }
  return quantized_empty_affine(shape, input.scalar_type(), input.layout(), input.device(), false,
      input.q_scale(), input.q_zero_point(), quantized_pool_format<Dimensions>(input));
}

template<bool Exact>
at::Tensor quantized_nearest2d(const at::Tensor& input, at::IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
  auto output = quantized_resize_output<2>(input, output_size);
  if (input.sizes() == output.sizes()) { quantized_copy_from(input, output, false); return output; }
  const auto sh = at::native::compute_scales_value<float>(scales_h, input.size(2), output_size[0]);
  const auto sw = at::native::compute_scales_value<float>(scales_w, input.size(3), output_size[1]);
  scientific_call(570, {quantized_pool_view<2>(input), quantized_pool_view<2>(output)},
      {int64_t(Exact)}, {1.0, sh, sw});
  return output;
}

template<bool Exact>
at::Tensor quantized_nearest3d(const at::Tensor& input, at::IntArrayRef output_size,
    std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
  auto output = quantized_resize_output<3>(input, output_size);
  if (input.sizes() == output.sizes()) { quantized_copy_from(input, output, false); return output; }
  const auto sd = at::native::compute_scales_value<float>(scales_d, input.size(2), output_size[0]);
  const auto sh = at::native::compute_scales_value<float>(scales_h, input.size(3), output_size[1]);
  const auto sw = at::native::compute_scales_value<float>(scales_w, input.size(4), output_size[2]);
  scientific_call(570, {quantized_pool_view<3>(input), quantized_pool_view<3>(output)},
      {int64_t(Exact)}, {sd, sh, sw});
  return output;
}

at::Tensor quantized_bilinear2d(const at::Tensor& input, at::IntArrayRef output_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
  auto output = quantized_resize_output<2>(input, output_size);
  if (!input.is_contiguous(at::MemoryFormat::ChannelsLast) && input.sizes() == output.sizes()) {
    quantized_copy_from(input, output, false); return output;
  }
  const auto sh = at::native::area_pixel_compute_scale<float>(input.size(2), output_size[0], align_corners, scales_h);
  const auto sw = at::native::area_pixel_compute_scale<float>(input.size(3), output_size[1], align_corners, scales_w);
  const auto ratio = static_cast<float>(output.q_scale() / input.q_scale());
  scientific_call(571, {quantized_pool_view<2>(input), quantized_pool_view<2>(output)},
      {int64_t(align_corners), input.q_zero_point()}, {sh, sw, ratio});
  return output;
}

const at::Tensor& quantized_resize_(const at::Tensor& self, at::IntArrayRef size,
    std::optional<at::MemoryFormat> format) {
  at::globalContext().alertNotDeterministic("quantized_resize_");
  TORCH_CHECK(!format.has_value(), "Unsupported memory format for quantized tensor resize");
  const auto scheme = self.qscheme();
  TORCH_CHECK(scheme == c10::kPerTensorAffine || scheme == c10::kPerTensorSymmetric,
              "Can only resize quantized tensors with per-tensor schemes!");
  quantized_storage_type(self.scalar_type());
  return resize_tensor(self, size, std::nullopt);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("upsample_nearest2d", TORCH_FN(quantized_nearest2d<false>));
  m.impl("_upsample_nearest_exact2d", TORCH_FN(quantized_nearest2d<true>));
  m.impl("upsample_nearest3d", TORCH_FN(quantized_nearest3d<false>));
  m.impl("_upsample_nearest_exact3d", TORCH_FN(quantized_nearest3d<true>));
  m.impl("upsample_bilinear2d", TORCH_FN(quantized_bilinear2d));
  m.impl("resize_", TORCH_FN(quantized_resize_));
}
