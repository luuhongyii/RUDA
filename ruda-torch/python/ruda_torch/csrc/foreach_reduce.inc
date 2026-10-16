std::vector<at::Tensor> foreach_reduce_launch(at::TensorList self, int64_t kind, bool root,
                                             at::ScalarType dtype) {
  std::vector<int64_t> params{kind, int64_t(root), static_cast<int64_t>(self.size())};
  params.reserve(3 + self.size());
  std::vector<at::Tensor> result;
  result.reserve(self.size());
  std::vector<at::Tensor> descriptors;
  descriptors.reserve(2 * self.size());
  for (const auto& tensor : self) {
    const auto alignment = 4 * tensor.element_size();
    params.push_back(reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % alignment == 0);
    result.emplace_back(at::empty({}, tensor.options().dtype(dtype)));
    descriptors.emplace_back(tensor.as_strided({tensor.numel()}, {1}));
  }
  for (const auto& tensor : result) descriptors.emplace_back(tensor.as_strided({1}, {1}));
  scientific_call(454, descriptors, params, {});
  return result;
}

void foreach_reduce_dtype(std::optional<at::ScalarType> dtype, at::ScalarType input, const char* name) {
  if (!dtype.has_value()) return;
  TORCH_CHECK(at::isFloatingType(*dtype) || at::isComplexType(*dtype),
              name, ": dtype should be floating point or complex, but got ", *dtype);
  TORCH_CHECK(at::isComplexType(input) == at::isComplexType(*dtype), name, ": dtype should be ",
              at::isComplexType(input) ? "complex" : "real", " for ",
              at::isComplexType(input) ? "complex" : "real", " inputs, but got ", *dtype);
  TORCH_CHECK(at::promoteTypes(input, *dtype) == *dtype, name, ": the dtype of the input (", input,
              ") should be convertible without narrowing to the specified dtype (", *dtype, ")");
}

template <bool PowerSum>
std::vector<at::Tensor> foreach_reduce_norm(at::TensorList self, const at::Scalar& ord,
                                           std::optional<at::ScalarType> dtype) {
  double p;
  if (ord.isIntegral(false)) p = static_cast<double>(ord.toLong());
  else if (ord.isFloatingPoint()) p = ord.toDouble();
  else TORCH_CHECK(false, PowerSum ? "foreach_tensor_powsum_cuda expects ord to be integer or float"
                                  : "foreach_tensor_norm_cuda expects ord to be integer or float");
  at::native::check_foreach_api_restrictions(self);
  if constexpr (!PowerSum) {
    if (p == std::numeric_limits<double>::infinity()) {
      for (const auto& tensor : self) {
        TORCH_CHECK(tensor.numel() > 0,
                    "_foreach_norm cannot compute the infinity norm on an empty tensor because the operation does not have an identity");
      }
    }
  }
  const bool integral_or_complex = std::any_of(self.begin(), self.end(), [](const at::Tensor& tensor) {
    return at::isIntegralType(tensor.scalar_type(), true) || tensor.is_complex();
  });
  const bool supported_order = p == 1. || p == 2. ||
      (!PowerSum && (p == 0. || p == std::numeric_limits<double>::infinity()));
  if (!foreach_binary_fast({self}, {}, false) || integral_or_complex || !supported_order) {
    if constexpr (PowerSum) {
#if __has_include(<ATen/ops/_foreach_powsum_native.h>)
      return at::native::foreach_tensor_powsum_slow(self, ord, dtype);
#else
      static_assert(!PowerSum, "this PyTorch build does not provide _foreach_powsum");
#endif
    } else return at::native::foreach_tensor_norm_slow(self, ord, dtype);
  }
  foreach_reduce_dtype(dtype, self[0].scalar_type(),
                       PowerSum ? "_foreach_tensor_powsum_cuda" : "_foreach_tensor_norm_cuda");
  const auto output_dtype = dtype.value_or(self[0].scalar_type());
  TORCH_CHECK(output_dtype == at::kFloat || output_dtype == at::kDouble ||
              output_dtype == at::kHalf || output_dtype == at::kBFloat16,
              "unsupported foreach norm output dtype: ", output_dtype);
  const int64_t kind = p == 0. ? 1 : p == 1. ? 2 : p == 2. ? 3 : 4;
  return foreach_reduce_launch(self, kind, !PowerSum && p == 2., output_dtype);
}

std::vector<at::Tensor> foreach_reduce_max(at::TensorList self) {
  at::native::check_foreach_api_restrictions(self);
  TORCH_CHECK(std::all_of(self.begin(), self.end(), [](const at::Tensor& tensor) { return tensor.numel() > 0; }),
              "_foreach_max cannot compute the maximum of an empty tensor; max over zero elements is undefined.");
  if (!foreach_binary_fast({self}, {}, false)) return at::native::foreach_tensor_max_slow(self);
  TORCH_CHECK(!self[0].is_complex(), "foreach max is not implemented for complex tensors");
  return foreach_reduce_launch(self, 0, false, self[0].scalar_type());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_foreach_norm.Scalar", foreach_reduce_norm<false>);
#if __has_include(<ATen/ops/_foreach_powsum_native.h>)
  m.impl("_foreach_powsum.Scalar", foreach_reduce_norm<true>);
#endif
  m.impl("_foreach_max", foreach_reduce_max);
}
