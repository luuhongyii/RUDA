at::Tensor& quantized_fill_scalar(at::Tensor& self, const at::Scalar& value) {
  validate(self.device());
  TORCH_CHECK(!value.isComplex(), "quantized fill requires a real scalar");
  auto raw = quantized_storage_view(self);
  auto real = at::empty({}, raw.options().dtype(at::kFloat));
  real.fill_(value.toFloat());
  auto expanded = real.expand(self.sizes());
  const auto scheme = self.qscheme();
  if (scheme == at::kPerChannelAffine || scheme == at::kPerChannelSymmetric ||
      scheme == at::kPerChannelAffineFloatQParams) {
    quantized_affine_channel(expanded, self, self.q_per_channel_scales(), self.q_per_channel_zero_points(),
        self.q_per_channel_axis(), false, scheme == at::kPerChannelAffineFloatQParams);
  } else {
    quantized_affine_scalar(expanded, self, self.q_scale(), self.q_zero_point(), false);
  }
  return self;
}

at::Tensor& quantized_fill_tensor(at::Tensor& self, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports a zero-dimensional value tensor");
  return quantized_fill_scalar(self, value.item());
}

at::Tensor& quantized_zero(at::Tensor& self) {
  return quantized_fill_scalar(self, at::Scalar(0));
}

at::Tensor& quantized_masked_fill_scalar(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
  quantized_arithmetic_input(self);
  TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill only supports boolean masks");
  TORCH_CHECK(mask.device() == self.device(), "masked_fill requires mask and self on the same device");
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes)
    TORCH_WARN("Use of masked_fill_ on expanded tensors is deprecated. Please clone the tensor first.");
  at::assert_no_partial_overlap(self, mask);
  auto expanded = at::expand_inplace(self, mask, "masked_fill_");
  scientific_call(576, {*expanded, quantized_storage_view(self)}, {self.q_zero_point()},
      {self.q_scale(), value.toFloat()});
  return self;
}

at::Tensor& quantized_masked_fill_tensor(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a zero-dimensional value tensor");
  return quantized_masked_fill_scalar(self, mask, value.item());
}

at::Tensor& quantized_index_select_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out) {
  quantized_arithmetic_input(self);
  dim = at::maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select index must be a scalar or vector");
  TORCH_CHECK(self.dim() != 0 || index.numel() == 1, "index_select of a scalar requires exactly one index");
  TORCH_CHECK(index.scalar_type() == at::kInt || index.scalar_type() == at::kLong, "index_select requires int32 or int64 indices");
  TORCH_CHECK(index.device() == self.device() || index.device().is_cpu(), "index_select index must be on CPU or the input device");
  TORCH_CHECK(out.device() == self.device() && out.scalar_type() == self.scalar_type(),
      "index_select input and output must have matching dtype and device");
  at::assert_no_internal_overlap(out); at::assert_no_overlap(out, self); at::assert_no_overlap(out, index);
  auto shape = self.sizes().vec(); if (self.dim() > 0) shape[dim] = index.numel();
  at::native::resize_output(out, shape);
  auto source = quantized_storage_view(self), target = quantized_storage_view(out);
  auto indices = scientific_resolve(index.to(self.device())).reshape({index.numel()});
  if (self.dim() == 0) { source = source.reshape({1}); target = target.reshape({1}); }
  scientific_call(470, {source, indices, target}, {dim, 1}, {});
  return out;
}

at::Tensor quantized_index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  quantized_arithmetic_input(self);
  auto out = quantized_empty_affine({0}, self.scalar_type(), self.layout(), self.device(), false,
      self.q_scale(), self.q_zero_point(), at::MemoryFormat::Contiguous);
  quantized_index_select_out(self, dim, index, out);
  return out;
}

at::Tensor quantized_index(const at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices) {
  quantized_arithmetic_input(self);
  TORCH_CHECK_INDEX(!indices.empty(), "at least one index must be provided");
  auto raw = quantized_storage_view(self).contiguous(self.suggest_memory_format());
  auto info = index_metadata(raw, indices);
  for (auto& index : info.indices) index = scientific_resolve(index.to(self.device()));
  at::Tensor gathered;
  at::TensorIteratorConfig config;
  config.set_check_mem_overlap(false).check_all_same_dtype(false)
      .add_output(gathered).add_owned_const_input(info.source)
      .declare_static_dtype_and_device(raw.scalar_type(), raw.device());
  for (const auto& index : info.indices) config.add_owned_const_input(index);
  auto iterator = config.build();
  gathered = iterator.output();
  index_dispatch(info, gathered);
  auto out = quantized_empty_affine(gathered.sizes(), self.scalar_type(), self.layout(), self.device(), false,
      self.q_scale(), self.q_zero_point(), gathered.suggest_memory_format());
  scientific_call(577, {gathered, quantized_storage_view(out)}, {self.q_zero_point()}, {self.q_scale()});
  return out;
}

template<int Dim>
std::vector<int64_t> quantized_reflection_shape(const at::Tensor& self, at::IntArrayRef padding) {
  at::native::padding::check_valid_input<Dim>(self, padding);
  auto shape = self.sizes().vec(); bool positive = false;
  for (int i = 0; i < Dim; ++i) {
    const auto axis = self.dim() - 1 - i, width = self.size(axis);
    TORCH_CHECK(padding[2*i] < width && padding[2*i+1] < width,
        "reflection padding must be smaller than the input dimension");
    shape[axis] = width + padding[2*i] + padding[2*i+1];
    TORCH_CHECK(shape[axis] >= 0, "reflection padding produced a negative output dimension");
    positive |= shape[axis] > 0;
  }
  TORCH_CHECK(positive, "reflection padding requires at least one positive spatial output dimension");
  return shape;
}

template<int Dim>
at::Tensor& quantized_reflection_out(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out) {
  validate(self.device()); quantized_storage_type(self.scalar_type());
  TORCH_CHECK(out.device() == self.device() && out.scalar_type() == self.scalar_type(),
      "reflection padding input and output must have matching dtype and device");
  auto shape = quantized_reflection_shape<Dim>(self, padding);
  at::native::resize_output(out, shape);
  if constexpr (Dim == 1) {
    quantized_arithmetic_input(self);
    at::set_quantizer_(out, at::make_per_tensor_affine_quantizer(self.q_scale(), self.q_zero_point(), self.scalar_type()));
  }
  std::vector<int64_t> params(9, 0);
  std::copy(padding.begin(), padding.end(), params.begin()); params[6] = Dim; params[7] = 1;
  scientific_call(323, {quantized_pool_view<Dim>(self), quantized_pool_view<Dim>(out)}, params, {});
  return out;
}

template<int Dim>
at::Tensor quantized_reflection(const at::Tensor& self, at::IntArrayRef padding) {
  quantized_arithmetic_input(self);
  const auto shape = quantized_reflection_shape<Dim>(self, padding);
  auto out = quantized_empty_affine(shape, self.scalar_type(), self.layout(), self.device(), false,
      self.q_scale(), self.q_zero_point(), at::MemoryFormat::Contiguous);
  quantized_reflection_out<Dim>(self, padding, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("fill_.Scalar", TORCH_FN(quantized_fill_scalar));
  m.impl("fill_.Tensor", TORCH_FN(quantized_fill_tensor));
  m.impl("zero_", TORCH_FN(quantized_zero));
  m.impl("masked_fill_.Scalar", TORCH_FN(quantized_masked_fill_scalar));
  m.impl("masked_fill_.Tensor", TORCH_FN(quantized_masked_fill_tensor));
  m.impl("index_select", TORCH_FN(quantized_index_select));
  m.impl("index_select.out", TORCH_FN(quantized_index_select_out));
  m.impl("index.Tensor", TORCH_FN(quantized_index));
  m.impl("reflection_pad1d", TORCH_FN(quantized_reflection<1>));
  m.impl("reflection_pad1d.out", TORCH_FN(quantized_reflection_out<1>));
  m.impl("reflection_pad2d", TORCH_FN(quantized_reflection<2>));
  m.impl("reflection_pad2d.out", TORCH_FN(quantized_reflection_out<2>));
}
