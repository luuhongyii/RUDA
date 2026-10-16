struct RudaLinearPackedParams final : LinearPackedParamsBase {
  at::Tensor packed_, scales_, zeros_;
  std::optional<at::Tensor> bias_;
  int64_t columns_, inner_;
  bool half_, per_channel_ = false;
  double scale_ = 1.0;
  int64_t zero_ = 0;

  RudaLinearPackedParams(const at::Tensor& weight, std::optional<at::Tensor> bias, bool half)
      : columns_(weight.size(0)), inner_(weight.size(1)), half_(half) {
    const auto options = weight.options().dtype(half ? at::kHalf : at::kChar);
    packed_ = at::empty({(columns_ + 15) / 16, (inner_ + 15) / 16, 16, 16}, options);
    if (half_) {
      if (weight.gt(65504.0).any().item<bool>() || weight.lt(-65504.0).any().item<bool>()) {
        TORCH_WARN("FOUND weight out of range ");
      }
      scientific_call(565, {scientific_resolve(weight), packed_}, {columns_, inner_, 0}, {});
    } else {
      per_channel_ = weight.qscheme() == c10::kPerChannelAffine;
      if (per_channel_) {
        TORCH_CHECK(weight.q_per_channel_axis() == 0, "quantized Linear weight channel axis must be0");
        auto scales = weight.q_per_channel_scales(); auto zeros = weight.q_per_channel_zero_points();
        TORCH_CHECK(scales.device() == weight.device() && zeros.device() == weight.device(),
                    "quantized Linear weight qparams must be on its device");
        TORCH_CHECK(scales.numel() == columns_ && zeros.numel() == columns_, "Linear weight qparam size mismatch");
        scales_ = scientific_resolve(scales).to(at::kFloat).contiguous();
        zeros_ = scientific_resolve(zeros).to(at::kInt).contiguous();
      } else {
        scale_ = static_cast<float>(weight.q_scale()); zero_ = weight.q_zero_point();
        scales_ = at::full({1}, scale_, weight.options().dtype(at::kFloat));
        zeros_ = at::full({1}, zero_, weight.options().dtype(at::kInt));
      }
      scientific_call(565, {quantized_storage_view(weight), packed_}, {columns_, inner_, 0}, {});
    }
    set_bias(std::move(bias));
  }

  void set_bias(std::optional<at::Tensor> bias) override {
    if (bias && bias->defined()) {
      TORCH_CHECK(bias->device() == packed_.device() && bias->scalar_type() == at::kFloat,
                  "quantized Linear bias must be Float on the weight device");
      TORCH_CHECK(bias->dim() == 1 && bias->numel() == columns_, "quantized Linear bias must have one value per output");
      bias_ = scientific_resolve(*bias).contiguous();
    } else { bias_ = std::nullopt; }
  }

  std::optional<at::Tensor> bias() override { return bias_; }

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override {
    at::Tensor weight;
    if (half_) {
      weight = at::empty({columns_, inner_}, packed_.options().dtype(at::kFloat));
      scientific_call(565, {packed_, weight}, {columns_, inner_, 1}, {});
    } else {
      if (per_channel_) {
        weight = quantized_empty_channel({columns_, inner_}, scales_.to(at::kDouble), zeros_.to(at::kLong), 0,
            at::kQInt8, at::kStrided, packed_.device(), false, at::MemoryFormat::Contiguous);
      } else {
        weight = quantized_empty_affine({columns_, inner_}, at::kQInt8, at::kStrided, packed_.device(), false,
                                        scale_, zero_, at::MemoryFormat::Contiguous);
      }
      scientific_call(565, {packed_, quantized_storage_view(weight)}, {columns_, inner_, 1}, {});
    }
    return {weight, bias_};
  }

  std::vector<int64_t> output_shape(const at::Tensor& input) const {
    validate(input.device());
    TORCH_CHECK(input.device() == packed_.device(), "quantized Linear input and packed weight must share a device");
    TORCH_CHECK(input.dim() >= 2 && input.size(-1) == inner_, "quantized Linear input rank or last dimension mismatch");
    auto shape = input.sizes().vec(); shape.back() = columns_; return shape;
  }

  at::Tensor bias_tensor() const {
    return bias_ ? *bias_ : at::empty({0}, packed_.options().dtype(at::kFloat));
  }

  at::Tensor apply_activation(at::Tensor input, double output_scale, int64_t output_zero, int64_t activation, double slope = 0) {
    TORCH_CHECK(!half_, "FP16 packed weights are only used by dynamic Linear");
    quantized_arithmetic_input(input);
    TORCH_CHECK(input.scalar_type() == at::kQInt8 || input.scalar_type() == at::kQUInt8,
                "quantized Linear input must be qint8 or quint8");
    auto out = quantized_empty_affine(output_shape(input), input.scalar_type(), input.layout(), input.device(), false,
                                     output_scale, output_zero, at::MemoryFormat::Contiguous);
    scientific_call(566, {quantized_storage_view(input), packed_, scales_, zeros_, bias_tensor(), quantized_storage_view(out)},
        {columns_, inner_, input.q_zero_point(), output_zero, activation, 0}, {input.q_scale(), output_scale, slope});
    return out;
  }

  at::Tensor apply(at::Tensor input, double scale, int64_t zero) override {
    return apply_activation(std::move(input), scale, zero, 0);
  }
  at::Tensor apply_relu(at::Tensor input, double scale, int64_t zero) override {
    return apply_activation(std::move(input), scale, zero, 1);
  }

  at::Tensor& apply_out(const at::Tensor& input, double scale, int64_t zero, at::Tensor& out) override {
    TORCH_CHECK(out.is_quantized() && out.q_scale() == scale && out.q_zero_point() == zero,
                "Linear output quantization parameters must match the requested parameters");
    return quantized_pool_copy_result(apply(input, scale, zero), out);
  }
  at::Tensor& apply_relu_out(const at::Tensor& input, double scale, int64_t zero, at::Tensor& out) override {
    TORCH_CHECK(out.is_quantized() && out.q_scale() == scale && out.q_zero_point() == zero,
                "Linear output quantization parameters must match the requested parameters");
    return quantized_pool_copy_result(apply_relu(input, scale, zero), out);
  }

  at::Tensor float_result(at::Tensor input, bool relu, double input_scale, int64_t input_zero) {
    TORCH_CHECK(input.scalar_type() == at::kFloat, "dynamic Linear input must be Float");
    auto out = at::empty(output_shape(input), input.options());
    auto real = scientific_resolve(input);
    if (half_) {
      scientific_call(567, {real, packed_, bias_tensor(), out}, {columns_, inner_, int64_t(relu)}, {});
    } else {
      TORCH_CHECK(input_zero >= 0 && input_zero <= 255, "dynamic Linear activation zero point must be in[0,255]");
      scientific_call(566, {real, packed_, scales_, zeros_, bias_tensor(), out},
          {columns_, inner_, input_zero, 0, int64_t(relu), 1}, {input_scale, 1.0, 0.0});
    }
    return out;
  }

  at::Tensor dynamic_result(at::Tensor input, bool reduce_range, bool relu) {
    output_shape(input);
    TORCH_CHECK(input.scalar_type() == at::kFloat, "dynamic Linear input must be Float");
    double scale = 1.0; int64_t zero = 0;
    if (!half_ && input.numel() != 0) {
      std::tie(scale, zero) = at::native::_choose_qparams_per_tensor(input, reduce_range);
    }
    return float_result(std::move(input), relu, scale, zero);
  }

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false) override {
    return dynamic_result(std::move(input), reduce_range, false);
  }
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false) override {
    return dynamic_result(std::move(input), reduce_range, true);
  }

  at::Tensor& float_out(const at::Tensor& value, at::Tensor& out) {
    TORCH_CHECK(out.device() == value.device() && out.scalar_type() == at::kFloat,
                "dynamic Linear output must be Float on the input device");
    at::assert_no_internal_overlap(out);
    if (at::native::resize_output_check(out, value.sizes())) resize_tensor(out, value.sizes(), std::nullopt);
    copy_from(value, out, false); return out;
  }
  at::Tensor& apply_dynamic_out(const at::Tensor& input, at::Tensor& out, bool reduce_range) override {
    return float_out(apply_dynamic(input, reduce_range), out);
  }
  at::Tensor& apply_dynamic_relu_out(const at::Tensor& input, at::Tensor& out, bool reduce_range) override {
    return float_out(apply_dynamic_relu(input, reduce_range), out);
  }

  at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32(at::Tensor input, double scale, int64_t zero) override {
    TORCH_CHECK(!half_, "QDQ Linear requires int8 packed weight");
    return float_result(std::move(input), false, scale, zero);
  }
  at::Tensor apply_with_input_q_dq_qweight_dq_relu_output_fp32(at::Tensor input, double scale, int64_t zero) override {
    TORCH_CHECK(!half_, "QDQ Linear requires int8 packed weight");
    return float_result(std::move(input), true, scale, zero);
  }
};

template<bool Half>
c10::intrusive_ptr<LinearPackedParamsBase> quantized_linear_prepack(at::Tensor weight, std::optional<at::Tensor> bias) {
  validate(weight.device());
  TORCH_CHECK(weight.dim() == 2, "quantized Linear weight must be two-dimensional");
  if constexpr (Half) {
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "FP16 Linear packing expects Float weights");
  } else {
    TORCH_CHECK(weight.scalar_type() == at::kQInt8, "quantized Linear weight must be qint8");
    TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine || weight.qscheme() == c10::kPerChannelAffine,
                "quantized Linear requires affine integer weight qparams");
  }
  return c10::make_intrusive<RudaLinearPackedParams>(weight, std::move(bias), Half);
}

RudaLinearPackedParams& quantized_linear_params(const c10::intrusive_ptr<LinearPackedParamsBase>& packed) {
  auto* result = dynamic_cast<RudaLinearPackedParams*>(packed.get());
  TORCH_CHECK(result, "Linear weight must be packed from RUDA tensors using quantized::linear_prepack[_fp16]");
  return *result;
}

template<int Activation>
at::Tensor quantized_linear_entry(at::Tensor input, const c10::intrusive_ptr<LinearPackedParamsBase>& packed,
    double scale, int64_t zero) {
  return quantized_linear_params(packed).apply_activation(std::move(input), scale, zero, Activation);
}
at::Tensor quantized_linear_leaky_entry(at::Tensor input, const c10::intrusive_ptr<LinearPackedParamsBase>& packed,
    double scale, int64_t zero, double slope) {
  return quantized_linear_params(packed).apply_activation(std::move(input), scale, zero, 2, slope);
}
template<bool Relu>
at::Tensor quantized_linear_dynamic_entry(at::Tensor input, const c10::intrusive_ptr<LinearPackedParamsBase>& packed, bool reduce_range) {
  return quantized_linear_params(packed).dynamic_result(std::move(input), reduce_range, Relu);
}
template<bool Relu>
at::Tensor quantized_linear_qdq_entry(at::Tensor input, double scale, int64_t zero,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed) {
  auto& params = quantized_linear_params(packed);
  TORCH_CHECK(!params.half_, "QDQ Linear requires int8 packed weight");
  return params.float_result(std::move(input), Relu, scale, zero);
}
template<bool Relu>
at::Tensor quantized_linear_fp16_entry(at::Tensor input, const c10::intrusive_ptr<LinearPackedParamsBase>& packed) {
  auto& params = quantized_linear_params(packed);
  TORCH_CHECK(params.half_, "FP16 dynamic Linear requires FP16 packed weight");
  return params.dynamic_result(std::move(input), false, Relu);
}
at::Tensor quantized_linear_fp16_unpacked(at::Tensor input, const at::Tensor& weight, const std::optional<at::Tensor>& bias) {
  auto packed = quantized_linear_prepack<true>(weight, bias);
  return packed->apply_dynamic(std::move(input), false);
}

at::Tensor quantize_dynamic_ruda(const at::Tensor& input, c10::ScalarType dtype, bool reduce_range) {
  validate(input.device());
  return at::native::quantize_per_tensor_dynamic(input, dtype, reduce_range);
}
at::Tensor& quantize_dynamic_out_ruda(const at::Tensor& input, c10::ScalarType dtype, bool reduce_range, at::Tensor& out) {
  auto value = quantize_dynamic_ruda(input, dtype, reduce_range);
  if (value.is_quantized()) return quantized_pool_copy_result(value, out);
  TORCH_CHECK(out.device() == value.device() && out.scalar_type() == value.scalar_type(),
              "dynamic quantization output dtype and device mismatch");
  at::assert_no_internal_overlap(out);
  if (at::native::resize_output_check(out, value.sizes())) resize_tensor(out, value.sizes(), std::nullopt);
  copy_from(value, out, false); return out;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("linear_prepack", TORCH_FN(quantized_linear_prepack<false>));
  m.impl("linear", TORCH_FN(quantized_linear_entry<0>));
  m.impl("linear_relu", TORCH_FN(quantized_linear_entry<1>));
  m.impl("linear_leaky_relu", TORCH_FN(quantized_linear_leaky_entry));
  m.impl("linear_tanh", TORCH_FN(quantized_linear_entry<3>));
}
TORCH_LIBRARY_IMPL(quantized, PrivateUse1, m) {
  m.impl("linear_dynamic", TORCH_FN(quantized_linear_dynamic_entry<false>));
  m.impl("linear_relu_dynamic", TORCH_FN(quantized_linear_dynamic_entry<true>));
  m.impl("linear_with_input_q_dq_qweight_dq_output_fp32", TORCH_FN(quantized_linear_qdq_entry<false>));
  m.impl("linear_with_input_q_dq_qweight_dq_relu_output_fp32", TORCH_FN(quantized_linear_qdq_entry<true>));
  m.impl("linear_prepack_fp16", TORCH_FN(quantized_linear_prepack<true>));
  m.impl("linear_dynamic_fp16", TORCH_FN(quantized_linear_fp16_entry<false>));
  m.impl("linear_relu_dynamic_fp16", TORCH_FN(quantized_linear_fp16_entry<true>));
  m.impl("linear_dynamic_fp16_unpacked_weight", TORCH_FN(quantized_linear_fp16_unpacked));
}
TORCH_LIBRARY_IMPL(_quantized, QuantizedPrivateUse1, m) {
  m.impl("linear_prepack", TORCH_FN(quantized_linear_prepack<false>));
  m.impl("linear", TORCH_FN(quantized_linear_entry<0>));
}
TORCH_LIBRARY_IMPL(_quantized, PrivateUse1, m) {
  m.impl("linear_dynamic", TORCH_FN(quantized_linear_dynamic_entry<false>));
  m.impl("linear_prepack_fp16", TORCH_FN(quantized_linear_prepack<true>));
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("quantize_per_tensor_dynamic", TORCH_FN(quantize_dynamic_ruda));
  m.impl("quantize_per_tensor_dynamic.out", TORCH_FN(quantize_dynamic_out_ruda));
}
TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("quantize_per_tensor_dynamic.out", TORCH_FN(quantize_dynamic_out_ruda));
}
