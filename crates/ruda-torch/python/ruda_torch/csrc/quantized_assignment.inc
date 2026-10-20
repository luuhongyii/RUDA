at::Tensor& quantized_index_put_impl(at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices, const at::Tensor& value,
    bool accumulate, bool unsafe) {
  TORCH_CHECK_INDEX(indices.size() <= size_t(self.dim()), "too many indices for tensor of dimension ", self.dim());
  TORCH_CHECK(!value.is_quantized(), "index_put values for a quantized tensor must not be quantized");
  quantized_arithmetic_input(self);
  TORCH_CHECK(!accumulate, "index_put for quantized tensors only supports accumulate=False");
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes)
    TORCH_WARN("Use of index_put_ on expanded tensors is deprecated. Please clone the tensor first.");

  auto masked = at::native::canDispatchToMaskedFill(self, indices, value);
  if (std::get<0>(masked))
    return quantized_masked_fill_scalar(self, std::get<1>(masked), value.item());

  TORCH_CHECK(value.scalar_type() == at::kFloat, "quantized index_put requires Float values");
  TORCH_CHECK(value.device() == self.device() || (value.device().is_cpu() && value.dim() == 0),
      "index_put values must be on the input device, except a CPU scalar");
  at::assert_no_overlap(self, value);
  for (const auto& optional : indices) {
    const std::optional<at::Tensor> index = optional;
    if (index && index->defined()) at::assert_no_overlap(self, *index);
  }

  const bool deterministic = at::globalContext().deterministicAlgorithms();
  auto raw = quantized_storage_view(self);
  auto target = deterministic ? raw.contiguous() : raw;
  auto info = index_metadata(target, indices);
  auto real = scientific_resolve(value.to(self.device()));
  TORCH_CHECK(at::is_expandable_to(real.sizes(), info.source.sizes()),
      "index_put values cannot broadcast to the indexing result shape");
  real = real.expand(info.source.sizes());

  std::vector<at::Tensor> tensors{target, info.source, real};
  tensors.reserve(3 + info.indices.size());
  for (auto& index : info.indices) {
    auto compact = index.as_strided(info.replacement,
        index.strides().slice(info.before, info.replacement.size()));
    tensors.push_back(scientific_resolve(compact.to(self.device())));
  }
  std::vector<int64_t> params{self.q_zero_point(), int64_t(deterministic), int64_t(unsafe),
      info.before, int64_t(info.replacement.size())};
  params.insert(params.end(), info.sizes.begin(), info.sizes.end());
  for (auto stride : info.strides) params.push_back(stride / int64_t(self.element_size()));
  scientific_call(578, tensors, params, {self.q_scale()});
  if (real.numel() != 0 && !target.is_same(raw)) copy_from(target, raw, false);
  return self;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("_index_put_impl_", TORCH_FN(quantized_index_put_impl));
}
