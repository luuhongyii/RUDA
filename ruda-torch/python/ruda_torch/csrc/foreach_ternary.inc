template <int Op> struct ForeachPointwiseSlow;
#define RUDA_FOREACH_POINTWISE_SLOW(ID, NAME) \
  template <> struct ForeachPointwiseSlow<ID> { \
    static std::vector<at::Tensor> scalar(at::TensorList self, at::TensorList first, at::TensorList second, \
                                        const at::Scalar& value, bool inplace) { \
      if (inplace) { at::native::foreach_tensor_##NAME##_scalar_slow_(self, first, second, value); return {}; } \
      return at::native::foreach_tensor_##NAME##_scalar_slow(self, first, second, value); \
    } \
    static std::vector<at::Tensor> scalars(at::TensorList self, at::TensorList first, at::TensorList second, \
                                         at::ArrayRef<at::Scalar> values, bool inplace) { \
      if (inplace) { at::native::foreach_tensor_##NAME##_scalarlist_slow_(self, first, second, values); return {}; } \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(self, first, second, values); \
    } \
  };
RUDA_FOREACH_POINTWISE_SLOW(0, addcmul)
RUDA_FOREACH_POINTWISE_SLOW(1, addcdiv)
#undef RUDA_FOREACH_POINTWISE_SLOW

bool foreach_ternary_integral(at::TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const at::Tensor& tensor) {
    return at::isIntegralType(tensor.scalar_type(), true);
  });
}

bool foreach_ternary_0d(at::TensorList tensors, const at::Tensor& reference) {
  return std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& tensor) {
    return tensor.dim() == 0 && tensor.scalar_type() == reference.scalar_type() &&
           tensor.device() == reference.device() && !foreach_copy_logical(tensor);
  });
}

std::vector<at::Tensor> foreach_ternary_launch(int64_t op, int64_t mode, bool inplace,
    at::TensorList self, at::TensorList first, at::TensorList second, at::ArrayRef<at::Scalar> scalars) {
  std::vector<int64_t> params{op, mode, static_cast<int64_t>(self.size())};
  params.reserve(3 + 2 * self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    foreach_binary_scalar_bits(params, mode == 3 ? at::Scalar(0) : scalars[scalars.size() == 1 ? 0 : i],
                               self[0].scalar_type());
  }
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  for (const auto& tensor : self) result.emplace_back(inplace ? tensor : at::native::empty_like(tensor));
  std::vector<at::Tensor> descriptors;
  descriptors.reserve(4 * self.size());
  for (const auto& list : std::vector<at::TensorList>{self, first, second, result}) {
    for (const auto& tensor : list) descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  }
  scientific_call(453, descriptors, params, {});
  if (inplace) {
    if (op != 2 || mode == 3) {
      for (const auto& tensor : self) tensor.unsafeGetTensorImpl()->bump_version();
    }
    return {};
  }
  return result;
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_pointwise_scalar(at::TensorList self, at::TensorList first,
                                                at::TensorList second, const at::Scalar& value) {
  at::native::check_foreach_api_restrictions(self, first, second);
  if (foreach_ternary_integral(self)) return ForeachPointwiseSlow<Op>::scalar(self, first, second, value, Inplace);
  if (foreach_binary_fast({self, first, second}, {value}, false)) {
    return foreach_ternary_launch(Op, 0, Inplace, self, first, second, {value});
  }
  if (foreach_ternary_0d(first, self[0]) && foreach_binary_fast({self, second}, {value}, false)) {
    return foreach_ternary_launch(Op, 1, Inplace, self, first, second, {value});
  }
  if constexpr (Op == 0) {
    if (foreach_ternary_0d(second, self[0]) && foreach_binary_fast({self, first}, {value}, false)) {
      return foreach_ternary_launch(Op, 1, Inplace, self, second, first, {value});
    }
  }
  return ForeachPointwiseSlow<Op>::scalar(self, first, second, value, Inplace);
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_pointwise_scalarlist(at::TensorList self, at::TensorList first,
                                                    at::TensorList second, at::ArrayRef<at::Scalar> values) {
  at::native::check_foreach_api_restrictions(self, first, second, values);
  if (!foreach_binary_fast({self, first, second}, values, false) || foreach_ternary_integral(self)) {
    return ForeachPointwiseSlow<Op>::scalars(self, first, second, values, Inplace);
  }
  return foreach_ternary_launch(Op, 0, Inplace, self, first, second, values);
}

template <int Op, bool Inplace>
std::vector<at::Tensor> foreach_pointwise_tensor(at::TensorList self, at::TensorList first,
                                                at::TensorList second, const at::Tensor& packed_scalars) {
  const auto values = at::native::convert_tensor_to_scalar_list(packed_scalars, self.size());
  at::native::check_foreach_api_restrictions(self, first, second, values);
  const auto promotion_values = Inplace ? at::ArrayRef<at::Scalar>(values) : at::ArrayRef<at::Scalar>{};
  if (!foreach_binary_fast({self, first, second}, promotion_values, false) || foreach_ternary_integral(self)) {
    return ForeachPointwiseSlow<Op>::scalars(self, first, second, values, Inplace);
  }
  return foreach_ternary_launch(Op, 0, Inplace, self, first, second, values);
}

template <bool Inplace>
std::vector<at::Tensor> foreach_lerp_list(at::TensorList self, at::TensorList end, at::TensorList weights) {
  at::native::check_foreach_api_restrictions(self, end, weights);
  if (!foreach_binary_fast({self, end, weights}, {}, true)) {
    if constexpr (Inplace) { at::native::foreach_tensor_ternary_lerp_slow_(self, end, weights); return {}; }
    else return at::native::foreach_tensor_ternary_lerp_slow(self, end, weights);
  }
  return foreach_ternary_launch(2, 3, Inplace, self, end, weights, {});
}

template <bool Inplace>
std::vector<at::Tensor> foreach_lerp_scalar(at::TensorList self, at::TensorList end, const at::Scalar& weight) {
  at::native::check_foreach_api_restrictions(self, end);
  if (!foreach_binary_fast({self, end}, {}, true)) {
    if constexpr (Inplace) { at::native::foreach_tensor_lerp_list_kernel_slow_(self, end, weight); return {}; }
    else return at::native::foreach_tensor_lerp_list_kernel_slow(self, end, weight);
  }
  return foreach_ternary_launch(2, 2, Inplace, self, end, self, {weight});
}

template <bool Inplace>
std::vector<at::Tensor> foreach_lerp_scalarlist(at::TensorList self, at::TensorList end, at::ArrayRef<at::Scalar> weights) {
  at::native::check_foreach_api_restrictions(self, end, weights);
  if (!foreach_binary_fast({self, end}, weights, true)) {
    if constexpr (Inplace) { at::native::foreach_tensor_lerp_scalarlist_kernel_slow_(self, end, weights); return {}; }
    else return at::native::foreach_tensor_lerp_scalarlist_kernel_slow(self, end, weights);
  }
  return foreach_ternary_launch(2, 2, Inplace, self, end, self, weights);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  #define RUDA_FOREACH_POINTWISE_REGISTER(ID, NAME) \
    m.impl("_foreach_" #NAME ".Scalar", [](at::TensorList self, at::TensorList first, at::TensorList second, const at::Scalar& value) { return foreach_pointwise_scalar<ID, false>(self, first, second, value); }); \
    m.impl("_foreach_" #NAME "_.Scalar", [](at::TensorList self, at::TensorList first, at::TensorList second, const at::Scalar& value) { foreach_pointwise_scalar<ID, true>(self, first, second, value); }); \
    m.impl("_foreach_" #NAME ".ScalarList", [](at::TensorList self, at::TensorList first, at::TensorList second, at::ArrayRef<at::Scalar> values) { return foreach_pointwise_scalarlist<ID, false>(self, first, second, values); }); \
    m.impl("_foreach_" #NAME "_.ScalarList", [](at::TensorList self, at::TensorList first, at::TensorList second, at::ArrayRef<at::Scalar> values) { foreach_pointwise_scalarlist<ID, true>(self, first, second, values); }); \
    m.impl("_foreach_" #NAME ".Tensor", [](at::TensorList self, at::TensorList first, at::TensorList second, const at::Tensor& values) { return foreach_pointwise_tensor<ID, false>(self, first, second, values); }); \
    m.impl("_foreach_" #NAME "_.Tensor", [](at::TensorList self, at::TensorList first, at::TensorList second, const at::Tensor& values) { foreach_pointwise_tensor<ID, true>(self, first, second, values); });
  RUDA_FOREACH_POINTWISE_REGISTER(0, addcmul)
  RUDA_FOREACH_POINTWISE_REGISTER(1, addcdiv)
  #undef RUDA_FOREACH_POINTWISE_REGISTER
  m.impl("_foreach_lerp.List", foreach_lerp_list<false>);
  m.impl("_foreach_lerp_.List", [](at::TensorList self, at::TensorList end, at::TensorList weights) { foreach_lerp_list<true>(self, end, weights); });
  m.impl("_foreach_lerp.Scalar", foreach_lerp_scalar<false>);
  m.impl("_foreach_lerp_.Scalar", [](at::TensorList self, at::TensorList end, const at::Scalar& weight) { foreach_lerp_scalar<true>(self, end, weight); });
  m.impl("_foreach_lerp.ScalarList", foreach_lerp_scalarlist<false>);
  m.impl("_foreach_lerp_.ScalarList", [](at::TensorList self, at::TensorList end, at::ArrayRef<at::Scalar> weights) { foreach_lerp_scalarlist<true>(self, end, weights); });
}
