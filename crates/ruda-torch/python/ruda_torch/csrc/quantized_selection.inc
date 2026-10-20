bool selection_all_flags(const at::Tensor& flags) {
  validate(flags.device());
  if (flags.numel() == 0) return true;
  auto source = flags.contiguous().reshape({flags.numel()});
  auto result = at::empty({}, source.options().dtype(at::kBool));
  ordering_call(0, {source, result}, {7, 1, 0});
  return scalar(result).toBool();
}

bool selection_equal_payload(const at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK(self.device() == other.device() && self.scalar_type() == other.scalar_type());
  TORCH_CHECK(self.sizes() == other.sizes());
  if (self.numel() == 0) return true;
  auto flags = at::empty({(self.numel() + 255) / 256}, self.options().dtype(at::kBool));
  scientific_call(573, {scientific_resolve(self), scientific_resolve(other), flags}, {}, {});
  return selection_all_flags(flags);
}

bool selection_equal(const at::Tensor& self, const at::Tensor& other) {
  if (self.sizes() != other.sizes()) return false;
  if (self.numel() == 0) return true;
  if (self.is_alias_of(other) && self.storage_offset() == other.storage_offset() &&
      self.scalar_type() == other.scalar_type() && self.is_contiguous() == other.is_contiguous() &&
      self.strides() == other.strides() && self.layout() == other.layout() &&
      self.is_neg() == other.is_neg() && self.is_conj() == other.is_conj()) return true;
  if (self.device() == other.device() && self.device().type() == c10::DeviceType::PrivateUse1 &&
      self.scalar_type() == other.scalar_type() && self.layout() == at::kStrided && other.layout() == at::kStrided &&
      (self.scalar_type() == at::kBool || self.scalar_type() == at::kByte || self.scalar_type() == at::kChar ||
       self.scalar_type() == at::kShort || self.scalar_type() == at::kInt || self.scalar_type() == at::kLong ||
       self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16 || self.scalar_type() == at::kFloat ||
       self.scalar_type() == at::kDouble)) return selection_equal_payload(self, other);
  return selection_all_flags(self.eq(other));
}

bool quantized_equal(const at::Tensor& self, const at::Tensor& other) {
  validate(self.device()); validate(other.device());
  TORCH_CHECK(self.device() == other.device(), "quantized equal requires tensors on the same device");
  if (!self.is_quantized() || !other.is_quantized()) return false;
  if (!self.quantizer()->equalTo(other.quantizer())) return false;
  if (self.sizes() != other.sizes() || self.scalar_type() != other.scalar_type()) return false;
  return selection_equal_payload(quantized_storage_view(self), quantized_storage_view(other));
}

std::tuple<at::Tensor, at::Tensor> quantized_topk(const at::Tensor& self, c10::SymInt k,
    int64_t dim, bool largest, bool sorted) {
  quantized_arithmetic_input(self);
  auto result = sorting_result(quantized_storage_view(self), 0, dim, k.expect_int(), largest, sorted, false);
  return {quantized_make_tensor(std::get<0>(result), self.q_scale(), self.q_zero_point()), std::get<1>(result)};
}

std::tuple<at::Tensor, at::Tensor> quantized_sort(const at::Tensor& self, std::optional<bool> stable,
    int64_t dim, bool descending) {
  quantized_arithmetic_input(self);
  auto input = quantized_storage_view(self).contiguous(self.suggest_memory_format());
  auto result = sorting_result(input, 1, dim, 0, descending, true, stable.value_or(false));
  return {quantized_make_tensor(std::get<0>(result), self.q_scale(), self.q_zero_point()), std::get<1>(result)};
}

template<bool Maximum>
std::tuple<at::Tensor, at::Tensor> quantized_extreme_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
  quantized_arithmetic_input(self);
  dim = at::maybe_wrap_dim(dim, self.dim(), true);
  const auto length = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK_INDEX(length > 0, "quantized extrema reduction has no identity for an empty dimension");
  const auto rows = self.numel() / length;
  auto shape = self.sizes().vec();
  if (self.dim() != 0) {
    if (keepdim) shape[dim] = 1;
    else shape.erase(shape.begin() + dim);
  }
  auto values = quantized_empty_affine(shape, self.scalar_type(), self.layout(), self.device(), false,
      self.q_scale(), self.q_zero_point(), at::MemoryFormat::Contiguous);
  auto indices = at::empty(shape, self.options().dtype(at::kLong));
  auto raw = quantized_storage_view(self);
  if (self.dim() != 0) {
    std::vector<int64_t> axes;
    for (int64_t axis = 0; axis < self.dim(); ++axis) if (axis != dim) axes.push_back(axis);
    axes.push_back(dim); raw = raw.permute(axes);
  }
  auto input = raw.contiguous().reshape({rows, length});
  ordering_call(0, {input, quantized_storage_view(values).reshape({rows}), indices.reshape({rows})},
      {Maximum ? 4 : 5, 2, 0});
  return {values, indices};
}

bool quantized_compare_fused(const at::Tensor& self, const at::Tensor& other) {
  return self.is_quantized() && other.is_quantized() && self.device() == other.device() &&
      self.qscheme() == c10::kPerTensorAffine && other.qscheme() == c10::kPerTensorAffine &&
      self.is_contiguous(self.suggest_memory_format()) && other.is_contiguous(other.suggest_memory_format());
}

template<int Operation>
at::Tensor quantized_compare_reference(const at::Tensor& self, const at::Tensor& other) {
  if constexpr (Operation == 0) return at::native::eq_quantized_cpu(self, other);
  else if constexpr (Operation == 1) return at::native::ne_quantized_cpu(self, other);
  else if constexpr (Operation == 2) return at::native::gt_quantized_cpu(self, other);
  else if constexpr (Operation == 3) return at::native::ge_quantized_cpu(self, other);
  else if constexpr (Operation == 4) return at::native::lt_quantized_cpu(self, other);
  else return at::native::le_quantized_cpu(self, other);
}

template<int Operation>
at::Tensor& quantized_compare_reference_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  if constexpr (Operation == 0) return at::native::eq_out_quantized_cpu(self, other, out);
  else if constexpr (Operation == 1) return at::native::ne_out_quantized_cpu(self, other, out);
  else if constexpr (Operation == 2) return at::native::gt_out_quantized_cpu(self, other, out);
  else if constexpr (Operation == 3) return at::native::ge_out_quantized_cpu(self, other, out);
  else if constexpr (Operation == 4) return at::native::lt_out_quantized_cpu(self, other, out);
  else return at::native::le_out_quantized_cpu(self, other, out);
}

template<int Operation>
at::Tensor quantized_compare_tensor(const at::Tensor& self, const at::Tensor& other) {
  if (!quantized_compare_fused(self, other)) return quantized_compare_reference<Operation>(self, other);
  auto a = quantized_storage_view(self); auto b = quantized_storage_view(other);
  auto output = at::empty({0}, self.options().dtype(at::kBool));
  auto iterator = at::TensorIteratorConfig().check_all_same_dtype(false).add_output(output)
      .add_const_input(a).add_const_input(b).build();
  output = iterator.output();
  scientific_call(572, {a.expand(output.sizes()), b.expand(output.sizes()), output},
      {Operation, self.q_zero_point(), other.q_zero_point()}, {self.q_scale(), other.q_scale()});
  return output;
}

template<int Operation>
at::Tensor& quantized_compare_tensor_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  at::infer_size_dimvector(self.sizes(), other.sizes());
  TORCH_CHECK(out.scalar_type() == at::kBool, "The 'out' tensor must have dtype 'torch.bool'");
  if (!quantized_compare_fused(self, other)) return quantized_compare_reference_out<Operation>(self, other, out);
  TORCH_CHECK(out.device() == self.device(), "comparison output must be on the input device");
  auto result = quantized_compare_tensor<Operation>(self, other);
  at::native::resize_output(out, result.sizes()); at::assert_no_internal_overlap(out);
  copy_from(result, out, false);
  return out;
}

#define RUDA_QUANTIZED_COMPARISON_SCALAR(name) \
at::Tensor quantized_##name##_scalar(const at::Tensor& self, const at::Scalar& other) { \
  return at::native::name##_quantized_cpu(self, other); \
} \
at::Tensor& quantized_##name##_scalar_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { \
  return at::native::name##_out_quantized_cpu(self, other, out); \
}
RUDA_QUANTIZED_COMPARISON_SCALAR(eq)
RUDA_QUANTIZED_COMPARISON_SCALAR(ne)
RUDA_QUANTIZED_COMPARISON_SCALAR(gt)
RUDA_QUANTIZED_COMPARISON_SCALAR(ge)
RUDA_QUANTIZED_COMPARISON_SCALAR(lt)
RUDA_QUANTIZED_COMPARISON_SCALAR(le)
#undef RUDA_QUANTIZED_COMPARISON_SCALAR

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("topk", TORCH_FN(quantized_topk));
  m.impl("sort.stable", TORCH_FN(quantized_sort));
  m.impl("max.dim", TORCH_FN(quantized_extreme_dim<true>));
  m.impl("min.dim", TORCH_FN(quantized_extreme_dim<false>));
  m.impl("max", TORCH_FN(at::native::max_quantized_cpu));
  m.impl("min", TORCH_FN(at::native::min_quantized_cpu));
  m.impl("max.unary_out", TORCH_FN(at::native::max_quantized_unary_out));
  m.impl("min.unary_out", TORCH_FN(at::native::min_quantized_unary_out));
  m.impl("equal", TORCH_FN(quantized_equal));
#define RUDA_QUANTIZED_COMPARISON(name, operation) \
  m.impl(#name ".Tensor", TORCH_FN(quantized_compare_tensor<operation>)); \
  m.impl(#name ".Tensor_out", TORCH_FN(quantized_compare_tensor_out<operation>)); \
  m.impl(#name ".Scalar", TORCH_FN(quantized_##name##_scalar)); \
  m.impl(#name ".Scalar_out", TORCH_FN(quantized_##name##_scalar_out));
  RUDA_QUANTIZED_COMPARISON(eq, 0)
  RUDA_QUANTIZED_COMPARISON(ne, 1)
  RUDA_QUANTIZED_COMPARISON(gt, 2)
  RUDA_QUANTIZED_COMPARISON(ge, 3)
  RUDA_QUANTIZED_COMPARISON(lt, 4)
  RUDA_QUANTIZED_COMPARISON(le, 5)
#undef RUDA_QUANTIZED_COMPARISON
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("equal", TORCH_FN(selection_equal));
}
