at::Tensor sparse_view_as_real(const at::Tensor& self) {
  TORCH_CHECK(self.is_sparse() && self.is_complex(), "view_as_real requires a complex sparse tensor");
  TORCH_CHECK(!self.is_conj(), "view_as_real requires resolving the conjugate tensor first");
  auto sizes = self.sym_sizes().vec();
  sizes.emplace_back(2);
  return at::_sparse_coo_tensor_with_dims_and_tensors_symint(
      self.sparse_dim(), self.dense_dim() + 1, sizes, self._indices(), at::view_as_real(self._values()),
      self.options().dtype(c10::toRealValueType(self.scalar_type())), self.is_coalesced());
}

at::Tensor sparse_view_as_complex(const at::Tensor& self) {
  TORCH_CHECK(self.is_sparse() && (self.scalar_type() == at::kHalf ||
      self.scalar_type() == at::kFloat || self.scalar_type() == at::kDouble),
      "view_as_complex requires a half, float or double sparse tensor");
  TORCH_CHECK(self.dense_dim() > 0 && self.size(-1) == 2,
      "view_as_complex requires a final dense dimension of size 2");
  auto sizes = self.sym_sizes().vec();
  sizes.pop_back();
  return at::_sparse_coo_tensor_with_dims_and_tensors_symint(
      self.sparse_dim(), self.dense_dim() - 1, sizes, self._indices(), at::view_as_complex(self._values()),
      self.options().dtype(c10::toComplexType(self.scalar_type())), self.is_coalesced());
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  m.impl("_sparse_coo_tensor_with_dims", TORCH_FN(at::native::new_with_dims_sparse));
  m.impl("_sparse_coo_tensor_with_dims_and_tensors", TORCH_FN(at::native::new_with_dims_and_tensor_sparse_symint));
  m.impl("_sparse_coo_tensor_with_dims.out", TORCH_FN(at::native::_sparse_coo_tensor_with_dims_out));
  m.impl("_sparse_coo_tensor_with_dims_and_tensors.out", TORCH_FN(at::native::_sparse_coo_tensor_with_dims_and_tensors_out_symint));
  m.impl("empty.memory_format", TORCH_FN(at::native::empty_sparse_symint));
  m.impl("empty_like", TORCH_FN(at::native::empty_like_sparse_coo));
  m.impl("clone", TORCH_FN(at::native::clone_sparse));
  m.impl("copy_", TORCH_FN(at::native::copy_sparse_wrapper_));
  m.impl("copy_sparse_to_sparse_", TORCH_FN(at::native::copy_sparse_));
  m.impl("sparse_resize_", TORCH_FN(at::native::sparse_resize_));
  m.impl("sparse_resize_and_clear_", TORCH_FN(at::native::sparse_resize_and_clear_));
  m.impl("resize_as_sparse_", TORCH_FN(at::native::resize_as_sparse_));
  m.impl("sparse_dim", TORCH_FN(at::native::sparse_dim_sparse));
  m.impl("_dimI", TORCH_FN(at::native::sparse_dim_sparse));
  m.impl("dense_dim", TORCH_FN(at::native::dense_dim_sparse));
  m.impl("_dimV", TORCH_FN(at::native::dense_dim_sparse));
  m.impl("_nnz", TORCH_FN(at::native::_nnz_sparse));
  m.impl("is_coalesced", TORCH_FN(at::native::is_coalesced_sparse));
  m.impl("_coalesced_", TORCH_FN(at::native::_coalesced_sparse_));
  m.impl("_indices", TORCH_FN(at::native::_indices_sparse));
  m.impl("_values", TORCH_FN(at::native::_values_sparse));
  m.impl("indices", TORCH_FN(at::native::indices_sparse));
  m.impl("values", TORCH_FN(at::native::values_sparse));
  m.impl("zero_", TORCH_FN(at::native::zero_sparse_));
  m.impl("zeros.out", TORCH_FN(at::native::zeros_sparse_out));
  m.impl("is_pinned", TORCH_FN(at::native::is_pinned_sparse_coo));
  m.impl("add.Tensor", TORCH_FN(at::native::add_sparse));
  m.impl("add_.Tensor", TORCH_FN(at::native::add_sparse_));
  m.impl("mul.Tensor", TORCH_FN(at::native::mul_sparse));
  m.impl("mul_.Tensor", TORCH_FN(at::native::mul_sparse_));
  m.impl("sub.Tensor", TORCH_FN(at::native::sub_sparse));
  m.impl("sub_.Tensor", TORCH_FN(at::native::sub_sparse_));
  m.impl("sub.out", TORCH_FN(at::native::sub_out_sparse));
  m.impl("neg", TORCH_FN(at::native::neg_sparse));
  m.impl("neg_", TORCH_FN(at::native::neg_sparse_));
  m.impl("neg.out", TORCH_FN(at::native::neg_out_sparse));
  m.impl("conj_physical.out", TORCH_FN(at::native::conj_physical_out_sparse));
#define RUDA_COO_UNARY(name) \
  m.impl(#name, TORCH_FN(at::native::name##_sparse)); \
  m.impl(#name "_", TORCH_FN(at::native::name##_sparse_)); \
  m.impl(#name ".out", TORCH_FN(at::native::name##_sparse_out));
  RUDA_COO_UNARY(abs)
  RUDA_COO_UNARY(asin)
  RUDA_COO_UNARY(asinh)
  RUDA_COO_UNARY(atan)
  RUDA_COO_UNARY(atanh)
  RUDA_COO_UNARY(ceil)
  RUDA_COO_UNARY(deg2rad)
  RUDA_COO_UNARY(erf)
  RUDA_COO_UNARY(erfinv)
  RUDA_COO_UNARY(expm1)
  RUDA_COO_UNARY(floor)
  RUDA_COO_UNARY(frac)
  RUDA_COO_UNARY(log1p)
  RUDA_COO_UNARY(nan_to_num)
  RUDA_COO_UNARY(rad2deg)
  RUDA_COO_UNARY(round)
  RUDA_COO_UNARY(sign)
  RUDA_COO_UNARY(sgn)
  RUDA_COO_UNARY(sin)
  RUDA_COO_UNARY(sinh)
  RUDA_COO_UNARY(sqrt)
  RUDA_COO_UNARY(tan)
  RUDA_COO_UNARY(tanh)
  RUDA_COO_UNARY(trunc)
#undef RUDA_COO_UNARY
#define RUDA_COO_UNARY_OUT(name) \
  m.impl(#name, TORCH_FN(at::native::name##_sparse)); \
  m.impl(#name ".out", TORCH_FN(at::native::name##_sparse_out));
  RUDA_COO_UNARY_OUT(signbit)
  RUDA_COO_UNARY_OUT(isneginf)
  RUDA_COO_UNARY_OUT(isposinf)
#undef RUDA_COO_UNARY_OUT
  m.impl("relu", TORCH_FN(at::native::relu_sparse));
  m.impl("relu_", TORCH_FN(at::native::relu_sparse_));
  m.impl("isnan", TORCH_FN(at::native::isnan_sparse));
  m.impl("isinf", TORCH_FN(at::native::isinf_sparse));
  m.impl("threshold_backward", TORCH_FN(at::native::threshold_backward_sparse));
  m.impl("threshold_backward.grad_input", TORCH_FN(at::native::threshold_backward_sparse_out));
  m.impl("view_as_real", TORCH_FN(sparse_view_as_real));
  m.impl("view_as_complex", TORCH_FN(sparse_view_as_complex));
  m.impl("permute", TORCH_FN(at::native::permute_sparse_coo));
  m.impl("unsqueeze", TORCH_FN(at::native::unsqueeze_sparse));
  m.impl("_sparse_broadcast_to", TORCH_FN(at::native::sparse_broadcast_to));
  m.impl("cat", TORCH_FN(at::native::cat_sparse));
  m.impl("narrow_copy", TORCH_FN(at::native::narrow_copy_sparse));
  m.impl("mv", TORCH_FN(at::native::mv_sparse));
  m.impl("pow.Tensor_Scalar", TORCH_FN(at::native::pow_sparse_scalar));
  m.impl("pow.Tensor_Scalar_out", TORCH_FN(at::native::pow_out_sparse_scalar));
  m.impl("div.Tensor", [](const at::Tensor& self, const at::Tensor& other) {
    return at::native::div_sparse(self, other);
  });
  m.impl("div_.Tensor", [](at::Tensor& self, const at::Tensor& other) -> at::Tensor& {
    return at::native::div_sparse_(self, other);
  });
  m.impl("div.out", [](const at::Tensor& self, const at::Tensor& other, at::Tensor& out) -> at::Tensor& {
    return at::native::div_out_sparse_zerodim(self, other, out);
  });
  m.impl("div.Tensor_mode", [](const at::Tensor& self, const at::Tensor& other, std::optional<c10::string_view> mode) {
    return at::native::div_sparse(self, other, mode);
  });
  m.impl("div_.Tensor_mode", [](at::Tensor& self, const at::Tensor& other, std::optional<c10::string_view> mode) -> at::Tensor& {
    return at::native::div_sparse_(self, other, mode);
  });
  m.impl("div.out_mode", [](const at::Tensor& self, const at::Tensor& other, std::optional<c10::string_view> mode, at::Tensor& out) -> at::Tensor& {
    return at::native::div_out_sparse_zerodim(self, other, mode, out);
  });
  m.impl("floor_divide", TORCH_FN(at::native::floor_divide_sparse));
  m.impl("floor_divide_", TORCH_FN(at::native::floor_divide_sparse_));
  m.impl("floor_divide.out", TORCH_FN(at::native::floor_divide_out_sparse_zerodim));
  m.impl("sum", TORCH_FN(at::native::sum_coo));
  m.impl("sum.dim_IntList", TORCH_FN(at::native::sum_sparse_coo));
  m.impl("any", TORCH_FN(at::native::any_sparse));
  m.impl("norm.ScalarOpt_dim", TORCH_FN(at::native::sparse_norm));
  m.impl("norm.ScalarOpt_dim_dtype", TORCH_FN(at::native::sparse_dtype_norm));
  m.impl("native_norm", [](const at::Tensor& self, const at::Scalar& p) {
    return at::native::norm_sparse(self, p);
  });
  m.impl("native_norm.ScalarOpt_dim_dtype", [](const at::Tensor& self, const std::optional<at::Scalar>& p,
      at::IntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
    return at::native::norm_sparse(self, p, dim, keepdim, dtype);
  });
  m.impl("_to_sparse.sparse_dim", [](const at::Tensor& self, int64_t sparse_dim) {
    return at::native::sparse_coo_to_sparse(self, sparse_dim);
  });
  m.impl("_to_sparse", [](const at::Tensor& self, std::optional<at::Layout> layout,
      at::OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim) {
    return at::native::sparse_coo_to_sparse(self, layout, blocksize, dense_dim);
  });
  m.impl("_to_sparse_csr", TORCH_FN(at::native::coo_to_sparse_csr));
  m.impl("_to_sparse_csc", TORCH_FN(at::native::coo_to_sparse_csc));
  m.impl("_to_sparse_bsr", TORCH_FN(at::native::coo_to_sparse_bsr));
  m.impl("_to_sparse_bsc", TORCH_FN(at::native::coo_to_sparse_bsc));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_to_sparse_csr", TORCH_FN(at::native::dense_to_sparse_csr));
  m.impl("_to_sparse_csc", TORCH_FN(at::native::dense_to_sparse_csc));
  m.impl("_to_sparse_bsr", TORCH_FN(at::native::dense_to_sparse_bsr));
  m.impl("_to_sparse_bsc", TORCH_FN(at::native::dense_to_sparse_bsc));
}

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  m.impl("select.int", TORCH_FN(at::native::select_sparse_csr));
  m.impl("select_copy.int", TORCH_FN(at::native::select_copy_sparse_csr));
  m.impl("sum", TORCH_FN(at::native::sum_csr));
  m.impl("sum.out", [](const at::Tensor& self, std::optional<at::ScalarType> dtype, at::Tensor& out) -> at::Tensor& {
    return at::native::sum_out(self, dtype, out);
  });
  m.impl("sum.dim_IntList", TORCH_FN(at::native::sum_sparse_compressed));
#define RUDA_COMPRESSED_UNARY(name) \
  m.impl(#name, TORCH_FN(at::native::name##_sparse_csr)); \
  m.impl(#name "_", TORCH_FN(at::native::name##_sparse_csr_)); \
  m.impl(#name ".out", TORCH_FN(at::native::name##_sparse_csr_out));
  RUDA_COMPRESSED_UNARY(abs)
  RUDA_COMPRESSED_UNARY(asin)
  RUDA_COMPRESSED_UNARY(asinh)
  RUDA_COMPRESSED_UNARY(atan)
  RUDA_COMPRESSED_UNARY(atanh)
  RUDA_COMPRESSED_UNARY(ceil)
  RUDA_COMPRESSED_UNARY(deg2rad)
  RUDA_COMPRESSED_UNARY(erf)
  RUDA_COMPRESSED_UNARY(erfinv)
  RUDA_COMPRESSED_UNARY(expm1)
  RUDA_COMPRESSED_UNARY(floor)
  RUDA_COMPRESSED_UNARY(frac)
  RUDA_COMPRESSED_UNARY(log1p)
  RUDA_COMPRESSED_UNARY(neg)
  RUDA_COMPRESSED_UNARY(rad2deg)
  RUDA_COMPRESSED_UNARY(round)
  RUDA_COMPRESSED_UNARY(sign)
  RUDA_COMPRESSED_UNARY(sin)
  RUDA_COMPRESSED_UNARY(sinh)
  RUDA_COMPRESSED_UNARY(sgn)
  RUDA_COMPRESSED_UNARY(sqrt)
  RUDA_COMPRESSED_UNARY(tan)
  RUDA_COMPRESSED_UNARY(tanh)
  RUDA_COMPRESSED_UNARY(trunc)
#undef RUDA_COMPRESSED_UNARY
#define RUDA_COMPRESSED_UNARY_OUT(name) \
  m.impl(#name, TORCH_FN(at::native::name##_sparse_csr)); \
  m.impl(#name ".out", TORCH_FN(at::native::name##_sparse_csr_out));
  RUDA_COMPRESSED_UNARY_OUT(angle)
  RUDA_COMPRESSED_UNARY_OUT(isneginf)
  RUDA_COMPRESSED_UNARY_OUT(isposinf)
  RUDA_COMPRESSED_UNARY_OUT(signbit)
#undef RUDA_COMPRESSED_UNARY_OUT
  m.impl("_conj_physical", TORCH_FN(at::native::conj_physical_sparse_csr));
  m.impl("conj_physical_", TORCH_FN(at::native::conj_physical_sparse_csr_));
  m.impl("conj_physical.out", TORCH_FN(at::native::conj_physical_sparse_csr_out));
  m.impl("relu", TORCH_FN(at::native::relu_sparse_csr));
  m.impl("relu_", TORCH_FN(at::native::relu_sparse_csr_));
  m.impl("isnan", TORCH_FN(at::native::isnan_sparse_csr));
  m.impl("isinf", TORCH_FN(at::native::isinf_sparse_csr));
  m.impl("normal_", TORCH_FN(at::native::normal_sparse_csr_));
  m.impl("threshold_backward", TORCH_FN(at::native::threshold_backward_sparse_compressed));
  m.impl("threshold_backward.grad_input", TORCH_FN(at::native::threshold_backward_sparse_compressed_out));
  m.impl("empty.memory_format", TORCH_FN(at::native::empty_sparse_compressed_symint));
  m.impl("empty_like", TORCH_FN(at::native::empty_like_sparse_csr));
  m.impl("clone", TORCH_FN(at::native::clone_sparse_compressed));
  m.impl("copy_", TORCH_FN(at::native::copy_sparse_compressed_));
  m.impl("resize_", TORCH_FN(at::native::resize_sparse_csr_));
  m.impl("resize_as_sparse_", TORCH_FN(at::native::resize_as_sparse_compressed_));
  m.impl("sparse_dim", TORCH_FN(at::native::sparse_dim_sparse_csr));
  m.impl("dense_dim", TORCH_FN(at::native::dense_dim_sparse_csr));
  m.impl("_nnz", TORCH_FN(at::native::_nnz_sparse_csr));
  m.impl("values", TORCH_FN(at::native::values_sparse_csr));
  m.impl("crow_indices", TORCH_FN(at::native::crow_indices_sparse_csr));
  m.impl("col_indices", TORCH_FN(at::native::col_indices_sparse_csr));
  m.impl("ccol_indices", TORCH_FN(at::native::ccol_indices_sparse_csr));
  m.impl("row_indices", TORCH_FN(at::native::row_indices_sparse_csr));
  m.impl("zero_", TORCH_FN(at::native::zero_sparse_csr_));
  m.impl("fill_.Scalar", TORCH_FN(at::native::fill_sparse_csr_));
  m.impl("mul.Tensor", TORCH_FN(at::native::mul_sparse_csr));
  m.impl("mul_.Tensor", TORCH_FN(at::native::mul_sparse_csr_));
  m.impl("mul.out", TORCH_FN(at::native::mul_out_sparse_csr));
  m.impl("mul.Scalar", TORCH_FN(at::native::mul_scalar_sparse_csr));
  m.impl("mul_.Scalar", TORCH_FN(at::native::mul__scalar_sparse_csr));
  m.impl("sparse_mask", TORCH_FN(at::native::sparse_mask_sparse_compressed));
  m.impl("is_pinned", TORCH_FN(at::native::is_pinned_sparse_compressed));
  m.impl("_to_dense", TORCH_FN(at::native::sparse_compressed_to_dense));
  m.impl("_to_sparse.sparse_dim", [](const at::Tensor& self, int64_t sparse_dim) {
    return at::native::sparse_compressed_to_sparse(self, sparse_dim);
  });
  m.impl("_to_sparse", [](const at::Tensor& self, std::optional<at::Layout> layout,
      at::OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim) {
    const auto target = layout.value_or(at::kSparse);
    if ((self.layout() == at::kSparseCsr || self.layout() == at::kSparseCsc) &&
        (target == at::kSparseBsr || target == at::kSparseBsc)) {
      TORCH_CHECK(blocksize.has_value(), "compressed-to-block conversion requires blocksize");
      return target == at::kSparseBsr ? self._to_sparse_bsr(*blocksize, dense_dim)
                                    : self._to_sparse_bsc(*blocksize, dense_dim);
    }
    return at::native::sparse_compressed_to_sparse(self, layout, blocksize, dense_dim);
  });
  m.impl("_to_sparse_csr", TORCH_FN(at::native::sparse_compressed_to_sparse_csr));
  m.impl("_to_sparse_csc", TORCH_FN(at::native::sparse_compressed_to_sparse_csc));
}
