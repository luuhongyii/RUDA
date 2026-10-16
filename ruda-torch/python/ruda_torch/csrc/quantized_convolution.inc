template<int Dimensions>
struct RudaConvPackedParams final : ConvPackedParamsBase<Dimensions> {
  at::Tensor packed_, scales_, zeros_;
  std::optional<at::Tensor> bias_;
  std::vector<int64_t> weight_shape_;
  torch::List<int64_t> stride_, padding_, output_padding_, dilation_;
  int64_t groups_, input_channels_, output_channels_, kernel_size_;
  bool transpose_, per_channel_;
  double scale_ = 1.0;
  int64_t zero_ = 0;

  RudaConvPackedParams(const at::Tensor& weight, std::optional<at::Tensor> bias,
      torch::List<int64_t> stride, torch::List<int64_t> padding, torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation, int64_t groups, bool transpose)
      : weight_shape_(weight.sizes().vec()), stride_(std::move(stride)), padding_(std::move(padding)),
        output_padding_(std::move(output_padding)), dilation_(std::move(dilation)), groups_(groups),
        input_channels_(transpose ? weight.size(0) : weight.size(1) * groups),
        output_channels_(transpose ? weight.size(1) * groups : weight.size(0)),
        kernel_size_(c10::multiply_integers(weight.sizes().begin() + 2, weight.sizes().end())),
        transpose_(transpose), per_channel_(weight.qscheme() == c10::kPerChannelAffine) {
    const int64_t input_group = input_channels_ / groups_, output_group = output_channels_ / groups_;
    const int64_t inner = input_group * kernel_size_;
    packed_ = at::empty({groups_, (output_group + 15) / 16, (inner + 15) / 16, 16, 16}, weight.options().dtype(at::kChar));
    if (per_channel_) {
      auto scales = weight.q_per_channel_scales(); auto zeros = weight.q_per_channel_zero_points();
      TORCH_CHECK(scales.device() == weight.device() && zeros.device() == weight.device(),
                  "quantized convolution weight qparams must be on the weight device");
      TORCH_CHECK(scales.numel() == output_channels_ && zeros.numel() == output_channels_,
                  "quantized convolution weight qparam size mismatch");
      scales_ = scientific_resolve(scales).to(at::kFloat).contiguous();
      zeros_ = scientific_resolve(zeros).to(at::kInt).contiguous();
    } else {
      scale_ = static_cast<float>(weight.q_scale()); zero_ = weight.q_zero_point();
      scales_ = at::full({1}, scale_, weight.options().dtype(at::kFloat));
      zeros_ = at::full({1}, zero_, weight.options().dtype(at::kInt));
    }
    if (bias && bias->defined()) {
      TORCH_CHECK(bias->device() == weight.device() && bias->scalar_type() == at::kFloat,
                  "quantized convolution bias must be Float on the weight device");
      TORCH_CHECK(bias->dim() == 1 && bias->numel() == output_channels_, "convolution bias size mismatch");
      bias_ = scientific_resolve(*bias).contiguous();
    }
    scientific_call(568, {quantized_pool_view<Dimensions>(weight), packed_},
        {groups_, input_group, output_group, kernel_size_, int64_t(transpose_), 0}, {});
  }

  torch::List<int64_t> stride() const override { return stride_; }
  torch::List<int64_t> padding() const override { return padding_; }
  torch::List<int64_t> output_padding() const override { return output_padding_; }
  torch::List<int64_t> dilation() const override { return dilation_; }
  int64_t groups() const override { return groups_; }
  bool transpose() const override { return transpose_; }

  at::MemoryFormat format() const {
    return Dimensions == 2 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  }

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override {
    at::Tensor weight;
    if (per_channel_) {
      weight = quantized_empty_channel(weight_shape_, scales_.to(at::kDouble), zeros_.to(at::kLong), 0,
          at::kQInt8, at::kStrided, packed_.device(), false, format());
    } else if (transpose_) {
      auto base_shape = weight_shape_; std::swap(base_shape[0], base_shape[1]);
      auto base = quantized_empty_affine(base_shape, at::kQInt8, at::kStrided, packed_.device(), false,
                                         scale_, zero_, format());
      auto strides = base.strides().vec(); std::swap(strides[0], strides[1]);
      weight = base.as_strided(weight_shape_, strides);
    } else {
      weight = quantized_empty_affine(weight_shape_, at::kQInt8, at::kStrided, packed_.device(), false,
                                      scale_, zero_, format());
    }
    scientific_call(568, {packed_, quantized_pool_view<Dimensions>(weight)},
        {groups_, input_channels_ / groups_, output_channels_ / groups_, kernel_size_, int64_t(transpose_), 1}, {});
    return {weight, bias_};
  }

  std::vector<int64_t> output_shape(const at::Tensor& input) const {
    validate(input.device());
    TORCH_CHECK(input.device() == packed_.device(), "convolution input and packed weight must share a device");
    TORCH_CHECK(input.dim() == Dimensions + 2, "quantized convolution expects a batched input");
    TORCH_CHECK(input.size(1) == input_channels_, "quantized convolution input channel mismatch");
    auto shape = transpose_ ? at::native::conv_input_size(input.sizes(), weight_shape_, padding_.vec(),
        output_padding_.vec(), stride_.vec(), dilation_.vec(), groups_) :
        at::native::conv_output_size(input.sizes(), weight_shape_, padding_.vec(), stride_.vec(), dilation_.vec());
    for (int i = 2; i < Dimensions + 2; ++i) {
      TORCH_CHECK(shape[i] >= 0 && (!(transpose_ || input.size(0) > 0) || shape[i] > 0),
                  "quantized convolution output spatial dimension must be positive for nonempty batches");
    }
    return shape;
  }

  at::Tensor run(const at::Tensor& input, double output_scale, int64_t output_zero, bool relu,
      const std::optional<at::Tensor>& accum = std::nullopt) {
    quantized_arithmetic_input(input);
    TORCH_CHECK(input.scalar_type() == at::kQInt8 || input.scalar_type() == at::kQUInt8,
                "quantized convolution input must be qint8 or quint8");
    TORCH_CHECK(!accum || (Dimensions == 2 && !transpose_), "conv2d_add only supports nontransposed Conv2d");
    const auto shape = output_shape(input);
    auto out = quantized_empty_affine(shape, input.scalar_type(), input.layout(), input.device(), false,
                                     output_scale, output_zero, format());
    if (out.numel() == 0) return out;
    at::Tensor target;
    auto residual = at::empty({0}, input.options().dtype(quantized_storage_type(input.scalar_type())));
    int64_t accum_zero = 0; double accum_scale = 1.0;
    if (accum) {
      quantized_arithmetic_pair(input, *accum);
      TORCH_CHECK(accum->sizes() == out.sizes(), "conv2d_add accumulation tensor must match output shape");
      target = accum->contiguous(format());
      accum_zero = target.q_zero_point(); accum_scale = target.q_scale();
      residual = quantized_pool_view<Dimensions>(target);
    }
    auto b = bias_ ? *bias_ : at::empty({0}, packed_.options().dtype(at::kFloat));
    std::array<int64_t, 3> kernel{1, 1, 1}, step{1, 1, 1}, pad{0, 0, 0}, dilate{1, 1, 1};
    for (int i = 0; i < Dimensions; ++i) {
      const auto j = 3 - Dimensions + i;
      kernel[j] = weight_shape_[i + 2]; step[j] = stride_.get(i); pad[j] = padding_.get(i); dilate[j] = dilation_.get(i);
    }
    scientific_call(569, {quantized_pool_view<Dimensions>(input), packed_, scales_, zeros_, b, residual,
                         quantized_pool_view<Dimensions>(out)},
        {groups_, input.q_zero_point(), output_zero, accum_zero, int64_t(transpose_), int64_t(relu), int64_t(accum.has_value()),
         kernel[0], kernel[1], kernel[2], step[0], step[1], step[2], pad[0], pad[1], pad[2], dilate[0], dilate[1], dilate[2]},
        {input.q_scale(), output_scale, accum_scale});
    if (accum) { quantized_copy_from(out, target, false); return target; }
    return out;
  }

  at::Tensor apply(const at::Tensor& input, double scale, int64_t zero) override { return run(input, scale, zero, false); }
  at::Tensor apply_relu(const at::Tensor& input, double scale, int64_t zero) override { return run(input, scale, zero, true); }
  at::Tensor apply_dynamic(const at::Tensor& input, bool reduce_range) override {
    output_shape(input);
    TORCH_CHECK(input.scalar_type() == at::kFloat, "dynamic convolution input must be Float");
    double scale = 1.0; int64_t zero = 0;
    if (input.numel() != 0) std::tie(scale, zero) = at::native::_choose_qparams_per_tensor(input, reduce_range);
    auto quantized = quantize_tensor_ruda(input, scale, zero, at::kQUInt8);
    auto result = apply(quantized, scale, zero);
    return at::native::dequantize_quantized(result);
  }
};

template<int Dimensions>
c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>> quantized_conv_pack_impl(at::Tensor weight,
    std::optional<at::Tensor> bias, torch::List<int64_t> stride, torch::List<int64_t> padding,
    torch::List<int64_t> output_padding, torch::List<int64_t> dilation, int64_t groups, bool transpose) {
  validate(weight.device());
  TORCH_CHECK(weight.scalar_type() == at::kQInt8, "quantized convolution weight must be qint8");
  TORCH_CHECK(weight.dim() == Dimensions + 2, "quantized convolution weight rank mismatch");
  TORCH_CHECK(groups > 0, "convolution groups must be positive");
  TORCH_CHECK(stride.size() == Dimensions && padding.size() == Dimensions && output_padding.size() == Dimensions &&
              dilation.size() == Dimensions, "convolution parameter length mismatch");
  for (int i = 0; i < Dimensions; ++i) {
    TORCH_CHECK(stride.get(i) > 0 && dilation.get(i) > 0 && padding.get(i) >= 0 && output_padding.get(i) >= 0,
                "convolution stride/dilation must be positive and padding nonnegative");
    TORCH_CHECK(weight.size(i + 2) > 0, "convolution kernel dimensions must be positive");
  }
  TORCH_CHECK(weight.size(0) > 0 && weight.size(1) > 0 && weight.size(0) % groups == 0,
              "convolution weight channel sizes must be positive and compatible with groups");
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine || weight.qscheme() == c10::kPerChannelAffine,
              "quantized convolution requires affine integer weight qparams");
  if (weight.qscheme() == c10::kPerChannelAffine) {
    TORCH_CHECK(!transpose, "Per Channel Quantization is currently disabled for transposed conv");
    TORCH_CHECK(weight.q_per_channel_axis() == 0, "convolution weight channel axis must be0");
  }
  return c10::make_intrusive<RudaConvPackedParams<Dimensions>>(weight, std::move(bias), std::move(stride),
      std::move(padding), std::move(output_padding), std::move(dilation), groups, transpose);
}

template<int Dimensions>
c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>> quantized_conv_prepack(at::Tensor weight, std::optional<at::Tensor> bias,
    torch::List<int64_t> stride, torch::List<int64_t> padding, torch::List<int64_t> dilation, int64_t groups) {
  torch::List<int64_t> output_padding;
  for (int i = 0; i < Dimensions; ++i) output_padding.push_back(0);
  return quantized_conv_pack_impl<Dimensions>(std::move(weight), std::move(bias), std::move(stride), std::move(padding),
      std::move(output_padding), std::move(dilation), groups, false);
}
template<int Dimensions>
c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>> quantized_conv_transpose_prepack(at::Tensor weight,
    std::optional<at::Tensor> bias, torch::List<int64_t> stride, torch::List<int64_t> padding,
    torch::List<int64_t> output_padding, torch::List<int64_t> dilation, int64_t groups) {
  return quantized_conv_pack_impl<Dimensions>(std::move(weight), std::move(bias), std::move(stride), std::move(padding),
      std::move(output_padding), std::move(dilation), groups, true);
}

torch::List<int64_t> quantized_conv1d_parameter(const torch::List<int64_t>& value, int64_t fill) {
  TORCH_CHECK(!value.empty(), "Conv1d parameters must contain elements");
  return torch::List<int64_t>({fill, value.get(value.size() == 1 ? 0 : 1)});
}
c10::intrusive_ptr<ConvPackedParamsBase<2>> quantized_conv1d_prepack(at::Tensor weight, std::optional<at::Tensor> bias,
    torch::List<int64_t> stride, torch::List<int64_t> padding, torch::List<int64_t> dilation, int64_t groups) {
  if (weight.dim() == 3) weight = weight.unsqueeze(2);
  return quantized_conv_prepack<2>(std::move(weight), std::move(bias), quantized_conv1d_parameter(stride, 1),
      quantized_conv1d_parameter(padding, 0), quantized_conv1d_parameter(dilation, 1), groups);
}
c10::intrusive_ptr<ConvPackedParamsBase<2>> quantized_conv_transpose1d_prepack(at::Tensor weight, std::optional<at::Tensor> bias,
    torch::List<int64_t> stride, torch::List<int64_t> padding, torch::List<int64_t> output_padding,
    torch::List<int64_t> dilation, int64_t groups) {
  if (weight.dim() == 3) weight = weight.unsqueeze(2);
  return quantized_conv_transpose_prepack<2>(std::move(weight), std::move(bias), quantized_conv1d_parameter(stride, 1),
      quantized_conv1d_parameter(padding, 0), quantized_conv1d_parameter(output_padding, 0),
      quantized_conv1d_parameter(dilation, 1), groups);
}

template<int Dimensions>
RudaConvPackedParams<Dimensions>& quantized_conv_params(const c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>>& packed) {
  auto* result = dynamic_cast<RudaConvPackedParams<Dimensions>*>(packed.get());
  TORCH_CHECK(result, "convolution weight must be packed from RUDA tensors");
  return *result;
}
template<int Dimensions, bool Relu>
at::Tensor quantized_conv_entry(at::Tensor input, const c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>>& packed,
    double scale, int64_t zero) {
  return quantized_conv_params(packed).run(input, scale, zero, Relu);
}
template<bool Relu>
at::Tensor quantized_conv1d_entry(at::Tensor input, const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed,
    double scale, int64_t zero) {
  return quantized_conv_params(packed).run(input.unsqueeze(2), scale, zero, Relu).squeeze_(2);
}
template<int Dimensions, bool Relu>
at::Tensor quantized_conv_legacy(at::Tensor input, const c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>>& packed,
    torch::List<int64_t>, torch::List<int64_t>, torch::List<int64_t>, int64_t, double scale, int64_t zero) {
  TORCH_WARN_ONCE("Arguments [stride, padding, dilation, groups] in ops.quantized.conv", Dimensions,
                  "d have been removed, please update your model to remove these arguments.");
  return quantized_conv_params(packed).run(input, scale, zero, Relu);
}
template<bool Relu>
at::Tensor quantized_conv_add(at::Tensor input, at::Tensor accum, const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed,
    double scale, int64_t zero) {
  return quantized_conv_params(packed).run(input, scale, zero, Relu, accum);
}
template<int Dimensions>
at::Tensor quantized_conv_dynamic(at::Tensor input, const c10::intrusive_ptr<ConvPackedParamsBase<Dimensions>>& packed,
    bool reduce_range) {
  return quantized_conv_params(packed).apply_dynamic(input, reduce_range);
}
at::Tensor quantized_conv1d_dynamic(at::Tensor input, const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed, bool reduce_range) {
  return quantized_conv_params(packed).apply_dynamic(input.unsqueeze(2), reduce_range).squeeze_(2);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("conv_prepack", TORCH_FN(quantized_conv_prepack<2>));
  m.impl("conv1d_prepack", TORCH_FN(quantized_conv1d_prepack));
  m.impl("conv2d_prepack", TORCH_FN(quantized_conv_prepack<2>));
  m.impl("conv3d_prepack", TORCH_FN(quantized_conv_prepack<3>));
  m.impl("conv_transpose1d_prepack", TORCH_FN(quantized_conv_transpose1d_prepack));
  m.impl("conv_transpose2d_prepack", TORCH_FN(quantized_conv_transpose_prepack<2>));
  m.impl("conv_transpose3d_prepack", TORCH_FN(quantized_conv_transpose_prepack<3>));
  m.impl("conv1d", TORCH_FN(quantized_conv1d_entry<false>));
  m.impl("conv1d_relu", TORCH_FN(quantized_conv1d_entry<true>));
  m.impl("conv2d.new", TORCH_FN((quantized_conv_entry<2, false>)));
  m.impl("conv2d_relu.new", TORCH_FN((quantized_conv_entry<2, true>)));
  m.impl("conv3d.new", TORCH_FN((quantized_conv_entry<3, false>)));
  m.impl("conv3d_relu.new", TORCH_FN((quantized_conv_entry<3, true>)));
  m.impl("conv2d", TORCH_FN((quantized_conv_legacy<2, false>)));
  m.impl("conv2d_relu", TORCH_FN((quantized_conv_legacy<2, true>)));
  m.impl("conv3d", TORCH_FN((quantized_conv_legacy<3, false>)));
  m.impl("conv3d_relu", TORCH_FN((quantized_conv_legacy<3, true>)));
  m.impl("conv2d_add", TORCH_FN(quantized_conv_add<false>));
  m.impl("conv2d_add_relu", TORCH_FN(quantized_conv_add<true>));
  m.impl("conv_transpose1d", TORCH_FN(quantized_conv1d_entry<false>));
  m.impl("conv_transpose2d", TORCH_FN((quantized_conv_entry<2, false>)));
  m.impl("conv_transpose3d", TORCH_FN((quantized_conv_entry<3, false>)));
}
TORCH_LIBRARY_IMPL(quantized, PrivateUse1, m) {
  m.impl("conv1d_dynamic", TORCH_FN(quantized_conv1d_dynamic));
  m.impl("conv2d_dynamic", TORCH_FN(quantized_conv_dynamic<2>));
  m.impl("conv3d_dynamic", TORCH_FN(quantized_conv_dynamic<3>));
  m.impl("conv_transpose1d_dynamic", TORCH_FN(quantized_conv1d_dynamic));
  m.impl("conv_transpose2d_dynamic", TORCH_FN(quantized_conv_dynamic<2>));
  m.impl("conv_transpose3d_dynamic", TORCH_FN(quantized_conv_dynamic<3>));
}
TORCH_LIBRARY_IMPL(_quantized, QuantizedPrivateUse1, m) {
  m.impl("conv2d_prepack", TORCH_FN(quantized_conv_prepack<2>));
  m.impl("conv3d_prepack", TORCH_FN(quantized_conv_prepack<3>));
  m.impl("conv_transpose1d_prepack", TORCH_FN(quantized_conv_transpose1d_prepack));
  m.impl("conv_transpose2d_prepack", TORCH_FN(quantized_conv_transpose_prepack<2>));
  m.impl("conv_transpose3d_prepack", TORCH_FN(quantized_conv_transpose_prepack<3>));
  m.impl("conv2d", TORCH_FN((quantized_conv_entry<2, false>)));
  m.impl("conv2d_relu", TORCH_FN((quantized_conv_entry<2, true>)));
  m.impl("conv3d", TORCH_FN((quantized_conv_entry<3, false>)));
  m.impl("conv3d_relu", TORCH_FN((quantized_conv_entry<3, true>)));
  m.impl("conv_transpose1d", TORCH_FN(quantized_conv1d_entry<false>));
  m.impl("conv_transpose2d", TORCH_FN((quantized_conv_entry<2, false>)));
}
