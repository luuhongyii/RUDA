void quantized_arithmetic_input(const at::Tensor& value) {
  validate(value.device());
  TORCH_CHECK(value.is_quantized() && value.qscheme() == c10::kPerTensorAffine,
              "RUDA quantized arithmetic requires per-tensor affine quantization");
  quantized_storage_type(value.scalar_type());
}

void quantized_arithmetic_pair(const at::Tensor& a, const at::Tensor& b) {
  quantized_arithmetic_input(a); quantized_arithmetic_input(b);
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.device() == b.device(),
              "quantized arithmetic operands must have the same dtype and device");
}

std::pair<int64_t, int64_t> quantized_arithmetic_bounds(c10::ScalarType dtype) {
  switch (dtype) {
    case at::kQInt8: return {-128, 127};
    case at::kQUInt8: return {0, 255};
    case at::kQInt32: return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    default: TORCH_CHECK(false, "unsupported quantized arithmetic dtype");
  }
}

void quantized_arithmetic_output(const at::Tensor& a, const at::Tensor& out, at::IntArrayRef shape) {
  quantized_arithmetic_pair(a, out);
  TORCH_CHECK(!out.is_same(a) || out.sizes() == shape,
              "quantized output cannot resize an input for broadcasting");
  at::assert_no_internal_overlap(out);
  at::assert_no_partial_overlap(out, a);
}

template<int Kind>
at::Tensor quantized_binary_out(at::Tensor a, at::Tensor b, at::Tensor out) {
  quantized_arithmetic_pair(a, b);
  auto shape = at::infer_size_dimvector(a.sizes(), b.sizes());
  quantized_arithmetic_output(a, out, shape);
  quantized_arithmetic_output(b, out, shape);
  if (at::native::resize_output_check(out, shape)) resize_tensor(out, shape, std::nullopt);
  auto a_raw = quantized_storage_view(a).expand(shape);
  auto b_raw = quantized_storage_view(b).expand(shape);
  scientific_call(554, {a_raw, b_raw, quantized_storage_view(out)},
      {int64_t(Kind >= 2), int64_t(Kind % 2), a.q_zero_point(), b.q_zero_point(), out.q_zero_point()},
      {a.q_scale(), b.q_scale(), out.q_scale()});
  return out;
}

template<int Kind>
at::Tensor quantized_binary_result(at::Tensor a, at::Tensor b, double scale, int64_t zero) {
  quantized_arithmetic_pair(a, b);
  auto shape = Kind >= 2 ? at::infer_size_dimvector(a.sizes(), b.sizes()) : at::DimVector(a.sizes());
  auto out = quantized_empty_affine(shape, a.scalar_type(), a.layout(), a.device(), false,
                                   scale, zero, a.suggest_memory_format());
  return quantized_binary_out<Kind>(a, b, out);
}

struct QuantizedScalarPlan {
  double input_scale, output_scale;
  int64_t input_zero, output_zero, mode = 0, offset = 0;
};

template<int Kind>
QuantizedScalarPlan quantized_scalar_plan(const at::Tensor& a, const at::Scalar& scalar) {
  quantized_arithmetic_input(a);
  const auto [minimum, maximum] = quantized_arithmetic_bounds(a.scalar_type());
  const double scale = a.q_scale(); const int64_t zero = a.q_zero_point();
  const double value = scalar.toDouble();
  QuantizedScalarPlan plan{scale, scale, zero, zero};
  if constexpr (Kind >= 2) {
    if (value > 0.0) {
      plan.output_scale = value * scale;
    } else if (value == 0.0) {
      plan.output_scale = 1.0; plan.output_zero = 0; plan.mode = 2;
    } else {
      plan.output_scale = std::abs(value) * scale;
      plan.output_zero = maximum - (zero - minimum); plan.mode = 3;
    }
  } else {
    const int64_t offset = static_cast<int64_t>(std::nearbyint(value / scale));
    const int64_t shifted_zero = zero - offset;
    if (minimum > shifted_zero) {
      plan.output_scale = (double(maximum) - shifted_zero) / (double(maximum) - minimum) * scale;
      plan.output_zero = minimum; plan.mode = 1;
    } else if (maximum < shifted_zero) {
      plan.output_scale = (double(shifted_zero) - minimum) / (double(maximum) - minimum) * scale;
      plan.output_zero = maximum; plan.mode = 1;
    } else {
      plan.output_zero = shifted_zero;
    }
    if (plan.mode == 1) plan.offset = at::Scalar(offset).toInt();
  }
  return plan;
}

template<int Kind>
at::Tensor quantized_scalar_execute(at::Tensor a, at::Tensor out, const QuantizedScalarPlan& plan) {
  quantized_arithmetic_output(a, out, a.sizes());
  if (at::native::resize_output_check(out, a.sizes())) resize_tensor(out, a.sizes(), std::nullopt);
  scientific_call(555, {quantized_storage_view(a), quantized_storage_view(out)},
      {plan.mode, int64_t(Kind % 2), plan.input_zero, plan.output_zero, plan.offset},
      {plan.input_scale, plan.output_scale});
  at::set_quantizer_(out, at::make_per_tensor_affine_quantizer(plan.output_scale, plan.output_zero, a.scalar_type()));
  return out;
}

template<int Kind>
at::Tensor quantized_scalar_result(at::Tensor a, const at::Scalar& scalar) {
  const auto plan = quantized_scalar_plan<Kind>(a, scalar);
  auto out = quantized_empty_affine(a.sizes(), a.scalar_type(), a.layout(), a.device(), false,
                                   plan.output_scale, plan.output_zero, a.suggest_memory_format());
  return quantized_scalar_execute<Kind>(a, out, plan);
}

template<int Kind>
at::Tensor quantized_scalar_out(at::Tensor a, const at::Scalar& scalar, at::Tensor out) {
  const auto plan = quantized_scalar_plan<Kind>(a, scalar);
  return quantized_scalar_execute<Kind>(a, out, plan);
}

template<int Kind>
at::Tensor quantized_scalar_reverse(const at::Scalar& scalar, at::Tensor a) {
  return quantized_scalar_result<Kind>(a, scalar);
}

template<int Kind>
at::Tensor quantized_scalar_tensor(at::Tensor a, at::Tensor scalar) {
  return quantized_scalar_result<Kind>(a, scalar.item());
}

template<int Kind>
at::Tensor quantized_scalar_tensor_out(at::Tensor a, at::Tensor scalar, at::Tensor out) {
  return quantized_scalar_out<Kind>(a, scalar.item(), out);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("add", TORCH_FN(quantized_binary_result<0>));
  m.impl("add.out", TORCH_FN(quantized_binary_out<0>));
  m.impl("add.Scalar", TORCH_FN(quantized_scalar_result<0>));
  m.impl("add.Scalar2", [](at::Scalar scalar, at::Tensor a) { return quantized_scalar_reverse<0>(scalar, a); });
  m.impl("add.Scalar_out", TORCH_FN(quantized_scalar_out<0>));
  m.impl("add_relu", TORCH_FN(quantized_binary_result<1>));
  m.impl("add_relu.out", TORCH_FN(quantized_binary_out<1>));
  m.impl("add_relu.Scalar", TORCH_FN(quantized_scalar_result<1>));
  m.impl("add_relu.Scalar2", [](at::Scalar scalar, at::Tensor a) { return quantized_scalar_reverse<1>(scalar, a); });
  m.impl("add_relu.Scalar_out", TORCH_FN(quantized_scalar_out<1>));
  m.impl("add_out", TORCH_FN(quantized_binary_out<0>));
  m.impl("add_relu_out", TORCH_FN(quantized_binary_out<1>));
  m.impl("add_scalar", TORCH_FN(quantized_scalar_result<0>));
  m.impl("add_scalar_relu", TORCH_FN(quantized_scalar_result<1>));
  m.impl("add_scalar_out", TORCH_FN(quantized_scalar_out<0>));
  m.impl("add_scalar_relu_out", TORCH_FN(quantized_scalar_out<1>));
  m.impl("add_scalar.Tensor", TORCH_FN(quantized_scalar_tensor<0>));
  m.impl("add_scalar_relu.Tensor", TORCH_FN(quantized_scalar_tensor<1>));
  m.impl("add_scalar_out.Tensor", TORCH_FN(quantized_scalar_tensor_out<0>));
  m.impl("add_scalar_relu_out.Tensor", TORCH_FN(quantized_scalar_tensor_out<1>));
  m.impl("mul", TORCH_FN(quantized_binary_result<2>));
  m.impl("mul.out", TORCH_FN(quantized_binary_out<2>));
  m.impl("mul.Scalar", TORCH_FN(quantized_scalar_result<2>));
  m.impl("mul.Scalar2", TORCH_FN(quantized_scalar_reverse<2>));
  m.impl("mul.Scalar_out", TORCH_FN(quantized_scalar_out<2>));
  m.impl("mul_relu", TORCH_FN(quantized_binary_result<3>));
  m.impl("mul_relu.out", TORCH_FN(quantized_binary_out<3>));
  m.impl("mul_relu.Scalar", TORCH_FN(quantized_scalar_result<3>));
  m.impl("mul_relu.Scalar2", TORCH_FN(quantized_scalar_reverse<3>));
  m.impl("mul_relu.Scalar_out", TORCH_FN(quantized_scalar_out<3>));
  m.impl("mul_out", TORCH_FN(quantized_binary_out<2>));
  m.impl("mul_relu_out", TORCH_FN(quantized_binary_out<3>));
  m.impl("mul_scalar", TORCH_FN(quantized_scalar_result<2>));
  m.impl("mul_scalar_relu", TORCH_FN(quantized_scalar_result<3>));
  m.impl("mul_scalar_out", TORCH_FN(quantized_scalar_out<2>));
  m.impl("mul_scalar_relu_out", TORCH_FN(quantized_scalar_out<3>));
  m.impl("mul_scalar.Tensor", TORCH_FN(quantized_scalar_tensor<2>));
  m.impl("mul_scalar_relu.Tensor", TORCH_FN(quantized_scalar_tensor<3>));
  m.impl("mul_scalar_out.Tensor", TORCH_FN(quantized_scalar_tensor_out<2>));
  m.impl("mul_scalar_relu_out.Tensor", TORCH_FN(quantized_scalar_tensor_out<3>));
}
TORCH_LIBRARY_IMPL(_quantized, QuantizedPrivateUse1, m) {
  m.impl("add", TORCH_FN(quantized_binary_result<0>));
}
