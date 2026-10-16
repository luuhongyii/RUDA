template <int Op> struct ForeachBinarySlow;

#define RUDA_FOREACH_SLOW_COMMON(NAME) \
  static std::vector<at::Tensor> scalar(at::TensorList self, const at::Scalar& value, bool inplace) { \
    if (inplace) { at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(self, value); return {}; } \
    return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(self, value); \
  } \
  static std::vector<at::Tensor> scalars(at::TensorList self, at::ArrayRef<at::Scalar> values, bool inplace) { \
    if (inplace) { at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(self, values); return {}; } \
    return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(self, values); \
  }
#define RUDA_FOREACH_SLOW_ALPHA_LIST(ID, NAME) \
  template <> struct ForeachBinarySlow<ID> { \
    RUDA_FOREACH_SLOW_COMMON(NAME) \
    static std::vector<at::Tensor> list(at::TensorList self, at::TensorList other, const at::Scalar& alpha, bool inplace) { \
      if (inplace) { at::native::foreach_tensor_##NAME##_list_kernel_slow_(self, other, alpha); return {}; } \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(self, other, alpha); \
    } \
  };
#define RUDA_FOREACH_SLOW_LIST(ID, NAME) \
  template <> struct ForeachBinarySlow<ID> { \
    RUDA_FOREACH_SLOW_COMMON(NAME) \
    static std::vector<at::Tensor> list(at::TensorList self, at::TensorList other, const at::Scalar&, bool inplace) { \
      if (inplace) { at::native::foreach_tensor_##NAME##_list_kernel_slow_(self, other); return {}; } \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(self, other); \
    } \
  };
RUDA_FOREACH_SLOW_ALPHA_LIST(0, add)
RUDA_FOREACH_SLOW_ALPHA_LIST(1, sub)
RUDA_FOREACH_SLOW_LIST(2, mul)
RUDA_FOREACH_SLOW_LIST(3, div)
RUDA_FOREACH_SLOW_LIST(4, clamp_min)
RUDA_FOREACH_SLOW_LIST(5, clamp_max)
RUDA_FOREACH_SLOW_LIST(6, pow)
#undef RUDA_FOREACH_SLOW_LIST
#undef RUDA_FOREACH_SLOW_ALPHA_LIST
#undef RUDA_FOREACH_SLOW_COMMON

bool foreach_binary_fast(const std::vector<at::TensorList>& lists, at::ArrayRef<at::Scalar> scalars,
                         bool promote) {
  if (!at::native::check_fast_path_restrictions(lists, scalars, promote)) return false;
  for (const auto& list : lists) {
    if (std::any_of(list.begin(), list.end(), foreach_copy_logical)) return false;
  }
  switch (lists[0][0].scalar_type()) {
    case at::kFloat: case at::kHalf: case at::kBFloat16: case at::kDouble:
    case at::kBool: case at::kLong: case at::kInt: case at::kShort: case at::kChar: case at::kByte:
    case at::kComplexFloat: case at::kComplexDouble: return true;
    default: return false;
  }
}

void foreach_binary_scalar_bits(std::vector<int64_t>& params, const at::Scalar& scalar,
                                at::ScalarType dtype) {
  uint64_t real = 0, imag = 0;
  switch (dtype) {
    case at::kFloat: case at::kHalf: case at::kBFloat16: real = scalar_bits<float>(scalar); break;
    case at::kDouble: real = scalar_bits<double>(scalar); break;
    case at::kBool: real = scalar_bits<bool>(scalar); break;
    case at::kLong: real = scalar_bits<int64_t>(scalar); break;
    case at::kInt: real = scalar_bits<int32_t>(scalar); break;
    case at::kShort: real = scalar_bits<int16_t>(scalar); break;
    case at::kChar: real = scalar_bits<int8_t>(scalar); break;
    case at::kByte: real = scalar_bits<uint8_t>(scalar); break;
    case at::kComplexFloat: {
      const auto value = scalar.to<c10::complex<float>>();
      real = scalar_bits<float>(at::Scalar(value.real()));
      imag = scalar_bits<float>(at::Scalar(value.imag()));
      break;
    }
    case at::kComplexDouble: {
      const auto value = scalar.to<c10::complex<double>>();
      real = scalar_bits<double>(at::Scalar(value.real()));
      imag = scalar_bits<double>(at::Scalar(value.imag()));
      break;
    }
    default: TORCH_CHECK(false, "unsupported foreach binary dtype: ", dtype);
  }
  int64_t real_bits, imag_bits;
  std::memcpy(&real_bits, &real, sizeof(real));
  std::memcpy(&imag_bits, &imag, sizeof(imag));
  params.push_back(real_bits);
  params.push_back(imag_bits);
}

std::vector<at::Tensor> foreach_binary_launch(int64_t op, int64_t mode, bool reverse, bool inplace,
    at::TensorList self, at::TensorList other, at::ArrayRef<at::Scalar> scalars,
    const at::Scalar& alpha, const at::Tensor& device_scalar = {}) {
  const auto dtype = self[0].scalar_type();
  TORCH_CHECK(!(op == 4 || op == 5) || !self[0].is_complex(),
              "foreach clamp is not implemented for complex tensors");
  TORCH_CHECK(!reverse || dtype != at::kBool, "foreach scalar pow is not implemented for bool tensors");
  std::vector<int64_t> params{op, mode, int64_t(reverse), static_cast<int64_t>(self.size())};
  params.reserve(6 + 2 * self.size());
  foreach_binary_scalar_bits(params, alpha, dtype);
  for (size_t i = 0; i < self.size(); ++i) {
    foreach_binary_scalar_bits(params, mode == 1 ? scalars[scalars.size() == 1 ? 0 : i] : at::Scalar(0), dtype);
  }
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  for (const auto& tensor : self) result.emplace_back(inplace ? tensor : at::native::empty_like(tensor));
  std::vector<at::Tensor> descriptors;
  descriptors.reserve(3 * self.size());
  for (const auto& tensor : self) descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  for (size_t i = 0; i < self.size(); ++i) {
    const auto& tensor = mode == 0 ? other[i] : (mode == 2 ? device_scalar : self[i]);
    descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  }
  for (const auto& tensor : result) descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  scientific_call(452, descriptors, params, {});
  if (inplace) {
    for (const auto& tensor : self) tensor.unsafeGetTensorImpl()->bump_version();
    return {};
  }
  return result;
}

at::Scalar foreach_binary_reciprocal(const at::Scalar& scalar) {
  if (scalar.isFloatingPoint()) return at::Scalar(1. / scalar.toDouble());
  if (scalar.isIntegral(true)) return at::Scalar(1. / static_cast<double>(scalar.toLong()));
  if (scalar.isComplex()) return at::Scalar(1. / scalar.toComplexDouble());
  TORCH_CHECK(false, "division with ", scalar.type(), " not supported");
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_binary_list(at::TensorList self, at::TensorList other,
                                           const at::Scalar& alpha = 1) {
  at::native::check_foreach_api_restrictions(self, other);
  const std::vector<at::TensorList> lists{self, other};
  const std::vector<at::Scalar> scalars = Op <= 1 ? std::vector<at::Scalar>{alpha} : std::vector<at::Scalar>{};
  if (!foreach_binary_fast(lists, scalars, Op >= 3)) {
    return ForeachBinarySlow<Op>::list(self, other, alpha, Inplace);
  }
  return foreach_binary_launch(Op, 0, false, Inplace, self, other, {}, alpha);
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_binary_scalar(at::TensorList self, const at::Scalar& scalar) {
  at::native::check_foreach_api_restrictions(self);
  if constexpr (Op == 1) at::native::sub_check(self[0], scalar);
  const bool fast = foreach_binary_fast({self}, {scalar}, Op >= 3);
  if constexpr (Op == 3) {
    const auto reciprocal = foreach_binary_reciprocal(scalar);
    if (!fast) return ForeachBinarySlow<2>::scalar(self, reciprocal, Inplace);
    return foreach_binary_launch(2, 1, false, Inplace, self, {}, {reciprocal}, 1);
  } else {
    if (!fast) return ForeachBinarySlow<Op>::scalar(self, scalar, Inplace);
    return foreach_binary_launch(Op, 1, false, Inplace, self, {}, {scalar}, 1);
  }
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_binary_scalarlist(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  at::native::check_foreach_api_restrictions(self, scalars);
  if constexpr (Op == 1) {
    for (size_t i = 0; i < self.size(); ++i) at::native::sub_check(self[i], scalars[i]);
  }
  if (!foreach_binary_fast({self}, scalars, Op >= 3)) {
    return ForeachBinarySlow<Op>::scalars(self, scalars, Inplace);
  }
  return foreach_binary_launch(Op, 1, false, Inplace, self, {}, scalars, 1);
}

std::vector<at::Tensor> foreach_binary_reverse_pow(const at::Scalar& scalar, at::TensorList exponent) {
  at::native::check_foreach_api_restrictions(exponent);
  if (!foreach_binary_fast({exponent}, {}, false)) {
    return at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent);
  }
  return foreach_binary_launch(6, 1, true, false, exponent, {}, {scalar}, 1);
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_binary_tensor(at::TensorList self, const at::Tensor& scalar,
                                             const at::Scalar& alpha = 1) {
  if constexpr (Op != 0) {
    if (scalar.device().is_cpu()) return foreach_binary_scalar<Op, Inplace>(self, scalar.item());
  }
  at::native::check_foreach_api_restrictions(self);
  const std::vector<at::Scalar> scalars = Op == 0 ? std::vector<at::Scalar>{alpha} : std::vector<at::Scalar>{};
  if (!foreach_binary_fast({self}, scalars, Op == 3) || self[0].scalar_type() != scalar.scalar_type() ||
      foreach_copy_logical(scalar)) {
    if constexpr (Op == 0) {
      if constexpr (Inplace) { at::native::foreach_tensor_add_tensor_kernel_slow_(self, scalar, alpha); return {}; }
      else return at::native::foreach_tensor_add_tensor_kernel_slow(self, scalar, alpha);
    } else if constexpr (Op == 2) {
      if constexpr (Inplace) { at::native::foreach_tensor_mul_tensor_kernel_slow_(self, scalar); return {}; }
      else return at::native::foreach_tensor_mul_tensor_kernel_slow(self, scalar);
    } else {
      if constexpr (Inplace) { at::native::foreach_tensor_div_tensor_kernel_slow_(self, scalar); return {}; }
      else return at::native::foreach_tensor_div_tensor_kernel_slow(self, scalar);
    }
  }
  TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,
              "scalar tensor expected to be 0 dim but it has ", scalar.dim(), " dimensions and ", scalar.numel(), " elements.");
  TORCH_CHECK(self[0].device() == scalar.device(), "scalar tensor expected to be on ", self[0].device(),
              " but is on ", scalar.device());
  return foreach_binary_launch(Op, 2, false, Inplace, self, {}, {}, alpha, scalar);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  #define RUDA_FOREACH_BINARY_SCALARS(ID, NAME) \
    m.impl("_foreach_" #NAME ".Scalar", [](at::TensorList self, const at::Scalar& scalar) { return foreach_binary_scalar<ID, false>(self, scalar); }); \
    m.impl("_foreach_" #NAME "_.Scalar", [](at::TensorList self, const at::Scalar& scalar) { foreach_binary_scalar<ID, true>(self, scalar); }); \
    m.impl("_foreach_" #NAME ".ScalarList", [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) { return foreach_binary_scalarlist<ID, false>(self, scalars); }); \
    m.impl("_foreach_" #NAME "_.ScalarList", [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) { foreach_binary_scalarlist<ID, true>(self, scalars); });
  #define RUDA_FOREACH_BINARY_ALPHA_LIST(ID, NAME) \
    RUDA_FOREACH_BINARY_SCALARS(ID, NAME) \
    m.impl("_foreach_" #NAME ".List", [](at::TensorList self, at::TensorList other, const at::Scalar& alpha) { return foreach_binary_list<ID, false>(self, other, alpha); }); \
    m.impl("_foreach_" #NAME "_.List", [](at::TensorList self, at::TensorList other, const at::Scalar& alpha) { foreach_binary_list<ID, true>(self, other, alpha); });
  #define RUDA_FOREACH_BINARY_LIST(ID, NAME) \
    RUDA_FOREACH_BINARY_SCALARS(ID, NAME) \
    m.impl("_foreach_" #NAME ".List", [](at::TensorList self, at::TensorList other) { return foreach_binary_list<ID, false>(self, other); }); \
    m.impl("_foreach_" #NAME "_.List", [](at::TensorList self, at::TensorList other) { foreach_binary_list<ID, true>(self, other); });
  RUDA_FOREACH_BINARY_ALPHA_LIST(0, add)
  RUDA_FOREACH_BINARY_ALPHA_LIST(1, sub)
  RUDA_FOREACH_BINARY_LIST(2, mul)
  RUDA_FOREACH_BINARY_LIST(3, div)
  RUDA_FOREACH_BINARY_LIST(4, clamp_min)
  RUDA_FOREACH_BINARY_LIST(5, clamp_max)
  RUDA_FOREACH_BINARY_LIST(6, pow)
  #undef RUDA_FOREACH_BINARY_LIST
  #undef RUDA_FOREACH_BINARY_ALPHA_LIST
  #undef RUDA_FOREACH_BINARY_SCALARS
  m.impl("_foreach_pow.ScalarAndTensor", foreach_binary_reverse_pow);
  m.impl("_foreach_add.Tensor", [](at::TensorList self, const at::Tensor& scalar, const at::Scalar& alpha) { return foreach_binary_tensor<0, false>(self, scalar, alpha); });
  m.impl("_foreach_add_.Tensor", [](at::TensorList self, const at::Tensor& scalar, const at::Scalar& alpha) { foreach_binary_tensor<0, true>(self, scalar, alpha); });
  #define RUDA_FOREACH_BINARY_TENSOR(ID, NAME) \
    m.impl("_foreach_" #NAME ".Tensor", [](at::TensorList self, const at::Tensor& scalar) { return foreach_binary_tensor<ID, false>(self, scalar); }); \
    m.impl("_foreach_" #NAME "_.Tensor", [](at::TensorList self, const at::Tensor& scalar) { foreach_binary_tensor<ID, true>(self, scalar); });
  RUDA_FOREACH_BINARY_TENSOR(2, mul)
  RUDA_FOREACH_BINARY_TENSOR(3, div)
  #undef RUDA_FOREACH_BINARY_TENSOR
}
