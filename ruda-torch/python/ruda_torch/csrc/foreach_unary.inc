bool foreach_unary_fast(const std::vector<at::Tensor>& tensors) {
  at::native::check_foreach_api_restrictions(tensors);
  std::vector<at::TensorList> lists;
  lists.emplace_back(tensors);
  return at::native::check_fast_path_restrictions(lists);
}

void foreach_unary_bump(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) tensor.unsafeGetTensorImpl()->bump_version();
}

std::vector<at::Tensor> foreach_unary_slow(const std::vector<at::Tensor>& tensors,
                                        int64_t op, bool inplace) {
  at::native::check_foreach_api_restrictions(tensors);
  #define RUDA_FOREACH_UNARY_CASE(ID, NAME) \
    case ID: \
      if (inplace) { at::native::foreach_tensor_##NAME##_slow_(tensors); return {}; } \
      return at::native::foreach_tensor_##NAME##_slow(tensors);
  switch (op) {
    RUDA_FOREACH_UNARY_CASE(0, erfc)
    RUDA_FOREACH_UNARY_CASE(1, lgamma)
    RUDA_FOREACH_UNARY_CASE(2, trunc)
    RUDA_FOREACH_UNARY_CASE(3, floor)
    RUDA_FOREACH_UNARY_CASE(4, ceil)
    RUDA_FOREACH_UNARY_CASE(5, acos)
    RUDA_FOREACH_UNARY_CASE(6, asin)
    RUDA_FOREACH_UNARY_CASE(7, atan)
    RUDA_FOREACH_UNARY_CASE(8, cosh)
    RUDA_FOREACH_UNARY_CASE(9, tan)
    RUDA_FOREACH_UNARY_CASE(10, sin)
    RUDA_FOREACH_UNARY_CASE(11, sinh)
    RUDA_FOREACH_UNARY_CASE(12, exp)
    RUDA_FOREACH_UNARY_CASE(13, expm1)
    RUDA_FOREACH_UNARY_CASE(14, tanh)
    RUDA_FOREACH_UNARY_CASE(15, log)
    RUDA_FOREACH_UNARY_CASE(16, log10)
    RUDA_FOREACH_UNARY_CASE(17, log2)
    RUDA_FOREACH_UNARY_CASE(18, log1p)
    RUDA_FOREACH_UNARY_CASE(19, cos)
    RUDA_FOREACH_UNARY_CASE(20, sqrt)
    RUDA_FOREACH_UNARY_CASE(21, erf)
    RUDA_FOREACH_UNARY_CASE(22, sigmoid)
    RUDA_FOREACH_UNARY_CASE(23, round)
    RUDA_FOREACH_UNARY_CASE(24, frac)
    RUDA_FOREACH_UNARY_CASE(25, reciprocal)
    RUDA_FOREACH_UNARY_CASE(26, sign)
    RUDA_FOREACH_UNARY_CASE(27, rsqrt)
    RUDA_FOREACH_UNARY_CASE(28, neg)
    RUDA_FOREACH_UNARY_CASE(29, abs)
    case 30:
      TORCH_CHECK(inplace, "foreach zero backend expects in-place dispatch");
      at::native::foreach_tensor_zero_slow_(tensors);
      return {};
    default: TORCH_CHECK(false, "unknown foreach unary operation");
  }
  #undef RUDA_FOREACH_UNARY_CASE
}

bool foreach_copy_logical(const at::Tensor& tensor) {
  return tensor.is_conj() || tensor.is_neg() || tensor._is_zerotensor();
}

void foreach_copy_(at::TensorList self, at::TensorList src, bool non_blocking) {
  at::native::check_foreach_api_restrictions(self, src);
  std::vector<at::TensorList> lists{self, src};
  if (self[0].device() != src[0].device() ||
      !at::native::_check_tensors_share_device_and_dtype({self}) ||
      !at::native::_check_tensors_share_device_and_dtype({src}) ||
      !at::native::_check_tensors_share_sizes_and_strides(lists) ||
      std::any_of(self.begin(), self.end(), foreach_copy_logical) ||
      std::any_of(src.begin(), src.end(), foreach_copy_logical) ||
      self[0].scalar_type() == at::kComplexHalf || src[0].scalar_type() == at::kComplexHalf ||
      self[0].scalar_type() == at::ScalarType::Float4_e2m1fn_x2 || src[0].scalar_type() == at::ScalarType::Float4_e2m1fn_x2) {
    at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
    return;
  }
  dtype_code(self[0].scalar_type());
  dtype_code(src[0].scalar_type());
  if (!self[0].is_complex() && src[0].is_complex()) {
    TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
  }
  std::vector<at::Tensor> descriptors;
  descriptors.reserve(src.size() + self.size());
  for (const auto& tensor : src) descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  for (const auto& tensor : self) descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  scientific_call(451, descriptors, {static_cast<int64_t>(self.size())}, {});
  for (const auto& tensor : self) tensor.unsafeGetTensorImpl()->bump_version();
}

#if __has_include(<ATen/ops/_foreach_clone_native.h>)
std::vector<at::Tensor> foreach_clone(at::TensorList self, std::optional<at::MemoryFormat> memory_format) {
  at::native::check_foreach_api_restrictions(self);
  std::vector<at::TensorList> lists{self};
  if (!at::native::_check_tensors_share_device_and_dtype(lists) ||
      std::any_of(self.begin(), self.end(), foreach_copy_logical) ||
      self[0].scalar_type() == at::kComplexHalf) {
    return at::native::foreach_tensor_clone_slow(self, memory_format);
  }
  const auto format = memory_format.value_or(at::MemoryFormat::Preserve);
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  for (const auto& tensor : self) {
    if (format == at::MemoryFormat::Preserve && tensor.is_non_overlapping_and_dense()) {
      result.emplace_back(at::empty_strided(tensor.sizes(), tensor.strides(), tensor.options()));
    } else {
      result.emplace_back(at::native::empty_like(tensor, {}, {}, {}, {}, format));
    }
  }
  foreach_copy_(result, self, false);
  return result;
}
#endif
