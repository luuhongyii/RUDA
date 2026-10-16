int64_t quantized_embedding_rows(const at::Tensor& input) {
  TORCH_CHECK(input.dim() >= 1, "rowwise embedding weights must have at least one dimension");
  int64_t rows = 1;
  for (int64_t dim = 0; dim + 1 < input.dim(); ++dim) {
    TORCH_CHECK(input.size(dim) == 0 || rows <= std::numeric_limits<int64_t>::max() / input.size(dim),
                "embedding row count overflow");
    rows *= input.size(dim);
  }
  return rows;
}

at::Tensor quantized_embedding_byte_pack_impl(const at::Tensor& weight, const std::optional<at::Tensor>& minmax) {
  validate(weight.device());
  TORCH_CHECK(weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kHalf,
              "embedding_bag_byte_prepack expects Float or Half weight");
  const auto rows = quantized_embedding_rows(weight), width = weight.size(-1);
  TORCH_CHECK(width <= std::numeric_limits<int64_t>::max() - 8, "embedding packed width overflow");
  auto input = scientific_resolve(weight).contiguous().reshape({rows, width});
  auto ranges = at::empty({0}, weight.options());
  if (minmax) {
    TORCH_CHECK(minmax->device() == weight.device() && minmax->scalar_type() == weight.scalar_type(),
                "rowwise_min_max must have the weight's device and dtype");
    TORCH_CHECK(minmax->dim() == 2 && minmax->size(0) == weight.size(0) && minmax->size(1) == 2,
                "rowwise_min_max must have shape [weight.size(0), 2]");
    TORCH_CHECK(minmax->size(0) >= rows, "rowwise_min_max does not cover all flattened embedding rows");
    ranges = scientific_resolve(*minmax).contiguous();
  }
  auto shape = weight.sizes().vec(); shape.back() = width + 8;
  auto output = at::empty(shape, weight.options().dtype(at::kByte));
  scientific_call(579, {input, ranges, output.reshape({rows, width + 8})}, {int64_t(minmax.has_value())}, {});
  return output;
}

at::Tensor quantized_embedding_byte_prepack(const at::Tensor& weight) {
  return quantized_embedding_byte_pack_impl(weight, std::nullopt);
}

at::Tensor quantized_embedding_byte_prepack_minmax(const at::Tensor& weight, const at::Tensor& minmax) {
  return quantized_embedding_byte_pack_impl(weight, minmax);
}

at::Tensor quantized_embedding_byte_unpack(const at::Tensor& packed) {
  validate(packed.device());
  TORCH_CHECK(packed.scalar_type() == at::kByte, "embedding_bag_byte_unpack expects Byte packed weight");
  const auto rows = quantized_embedding_rows(packed);
  TORCH_CHECK(packed.size(-1) >= 8, "byte packed embedding row must contain eight scale/bias bytes");
  const auto width = packed.size(-1) - 8;
  auto shape = packed.sizes().vec(); shape.back() = width;
  auto output = at::empty(shape, packed.options().dtype(at::kFloat));
  scientific_call(581, {scientific_resolve(packed).contiguous().reshape({rows, width + 8}), output.reshape({rows, width})}, {}, {});
  return output;
}

at::Tensor quantized_embedding_byte_lookup(const at::Tensor& packed, const at::Tensor& indices,
    const std::optional<at::Tensor>& offsets, bool pruned,
    const std::optional<at::Tensor>& per_sample_weights, const std::optional<at::Tensor>& mapping,
    bool include_last, bool embedding) {
  validate(packed.device());
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 2 && packed.size(1) >= 8,
              "byte embedding weight must be a rank-two Byte tensor with scale/bias headers");
  TORCH_CHECK(indices.device() == packed.device() && (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong),
              "embedding indices must be Int or Long on the weight device");
  TORCH_CHECK(indices.dim() == 1 || indices.dim() == 2, "quantized embedding expects one- or two-dimensional indices");
  TORCH_CHECK(packed.is_contiguous() && indices.is_contiguous(), "embedding weight and indices must be contiguous");
  const auto length = indices.numel(), width = packed.size(1) - 8;
  auto flat_indices = scientific_resolve(indices).reshape({length});
  auto flat_offsets = at::empty({0}, indices.options());
  int64_t kind = 0, fixed = 1, bags = length;
  if (!embedding) {
    if (indices.dim() == 2) {
      TORCH_CHECK(!offsets, "embedding_bag with 2D indices requires offsets=None");
      kind = 2; fixed = indices.size(1);
      TORCH_CHECK(fixed > 0, "embedding_bag with 2D indices requires a nonzero sequence length");
      bags = indices.size(0) - int64_t(include_last);
    } else {
      TORCH_CHECK(offsets.has_value(), "embedding_bag with 1D indices requires offsets");
      TORCH_CHECK(offsets->device() == packed.device() && (offsets->scalar_type() == at::kInt || offsets->scalar_type() == at::kLong),
                  "embedding offsets must be Int or Long on the weight device");
      TORCH_CHECK(offsets->dim() == 1 && offsets->is_contiguous(), "embedding offsets must be contiguous and one-dimensional");
      kind = 1; bags = offsets->numel() - int64_t(include_last);
      flat_offsets = scientific_resolve(*offsets);
    }
  } else { include_last = false; }
  TORCH_CHECK(bags >= 0, "include_last_offset requires at least one offset");
  auto weights = at::empty({0}, packed.options().dtype(at::kFloat));
  if (per_sample_weights) {
    TORCH_CHECK(per_sample_weights->device() == packed.device() && per_sample_weights->scalar_type() == at::kFloat,
                "byte embedding per_sample_weights must be Float on the weight device");
    TORCH_CHECK(per_sample_weights->numel() == length, "embedding per_sample_weights must have one element per index");
    weights = scientific_resolve(*per_sample_weights).contiguous().reshape({length});
  }
  auto compressed = at::empty({0}, packed.options().dtype(at::kInt));
  if (pruned) {
    TORCH_CHECK(mapping.has_value(), "pruned embedding requires compressed_indices_mapping");
    TORCH_CHECK(mapping->device() == packed.device() && mapping->scalar_type() == at::kInt,
                "compressed_indices_mapping must be Int on the weight device");
    compressed = scientific_resolve(*mapping).contiguous().reshape({mapping->numel()});
  }
  const auto work = std::max<int64_t>(1, std::max(length, flat_offsets.numel()));
  auto flags = at::empty({(work + 255) / 256}, packed.options().dtype(at::kBool));
  std::vector<int64_t> params{kind, bags, fixed, int64_t(include_last), int64_t(pruned), int64_t(per_sample_weights.has_value())};
  scientific_call(582, {packed, flat_indices, flat_offsets, weights, compressed, flags}, params, {});
  TORCH_CHECK(selection_all_flags(flags), "quantized embedding has invalid offsets, indices or compressed row mapping");
  auto output = at::empty({bags, width}, packed.options().dtype(at::kFloat));
  scientific_call(583, {packed, flat_indices, flat_offsets, weights, compressed, output}, params, {});
  if (embedding && indices.dim() == 2) return output.reshape({indices.size(0), indices.size(1), width});
  return output;
}

struct RudaEmbeddingPackedParams final : EmbeddingPackedParamsBase {
  at::Tensor packed_, scales_, zeros_;
  int64_t rows_, width_;

  explicit RudaEmbeddingPackedParams(const at::Tensor& weight) : rows_(weight.size(0)), width_(weight.size(1)) {
    auto scales = weight.q_per_channel_scales(), zeros = weight.q_per_channel_zero_points();
    TORCH_CHECK(scales.device() == weight.device() && zeros.device() == weight.device(),
                "embedding weight qparams must be on the weight device");
    TORCH_CHECK(scales.scalar_type() == at::kFloat && zeros.scalar_type() == at::kFloat &&
                scales.numel() == rows_ && zeros.numel() == rows_, "embedding weight requires Float qparams for each row");
    scales_ = scientific_resolve(scales).reshape({rows_}).clone(at::MemoryFormat::Contiguous);
    zeros_ = scientific_resolve(zeros).reshape({rows_}).clone(at::MemoryFormat::Contiguous);
    TORCH_CHECK(width_ <= std::numeric_limits<int64_t>::max() - 8, "embedding packed width overflow");
    packed_ = at::empty({rows_, width_ + 8}, weight.options().dtype(at::kByte));
    scientific_call(580, {quantized_storage_view(weight), scales_, zeros_, packed_}, {}, {});
  }

  at::Tensor embeddingbag_byte(const at::Tensor& indices, const std::optional<at::Tensor>& offsets,
      bool pruned, const std::optional<at::Tensor>& weights, const std::optional<at::Tensor>& mapping,
      bool include_last, bool embedding) override {
    return quantized_embedding_byte_lookup(packed_, indices, offsets, pruned, weights, mapping, include_last, embedding);
  }

  at::Tensor embeddingbag_4bit(const at::Tensor&, const std::optional<at::Tensor>&, bool,
      const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, bool, bool) override {
    TORCH_CHECK(false, "RUDA byte-packed embedding weight cannot be used by a 4-bit embedding operator");
  }

  at::Tensor unpack() override {
    auto scales = scales_.clone(at::MemoryFormat::Contiguous), zeros = zeros_.clone(at::MemoryFormat::Contiguous);
    auto output = quantized_empty_channel({rows_, width_}, scales, zeros, 0, at::kQUInt8,
        at::kStrided, packed_.device(), false, at::MemoryFormat::Contiguous);
    copy_from(packed_.narrow(1, 0, width_), quantized_storage_view(output), false);
    return output;
  }

  int64_t bit_rate() const override { return 8; }
  int64_t version() const override { return 1; }
};

c10::intrusive_ptr<EmbeddingPackedParamsBase> quantized_embedding_prepack(const at::Tensor& weight) {
  validate(weight.device());
  TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == at::kQUInt8,
              "RUDA embedding_bag_prepack currently requires a rank-two quint8 weight");
  TORCH_CHECK(weight.qscheme() == c10::kPerChannelAffineFloatQParams && weight.q_per_channel_axis() == 0,
              "embedding weight requires per-channel affine Float qparams along axis zero");
  return c10::make_intrusive<RudaEmbeddingPackedParams>(weight);
}

RudaEmbeddingPackedParams& quantized_embedding_params(const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed) {
  auto* params = dynamic_cast<RudaEmbeddingPackedParams*>(packed.get());
  TORCH_CHECK(params, "RUDA embedding requires weights packed on RUDA");
  return *params;
}

at::Tensor quantized_embedding_bag_byte(const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed,
    const at::Tensor& indices, const std::optional<at::Tensor>& offsets, bool scale_grad_by_freq, int64_t mode,
    bool pruned, const std::optional<at::Tensor>& weights, const std::optional<at::Tensor>& mapping, bool include_last) {
  (void)scale_grad_by_freq; (void)mode;
  return quantized_embedding_params(packed).embeddingbag_byte(indices, offsets, pruned, weights, mapping, include_last, false);
}

at::Tensor quantized_embedding_byte(const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed,
    const at::Tensor& indices, bool pruned) {
  return quantized_embedding_params(packed).embeddingbag_byte(indices, std::nullopt, pruned, std::nullopt, std::nullopt, false, true);
}

at::Tensor quantized_embedding_byte_rowwise(const at::Tensor& weight, const at::Tensor& indices,
    const std::optional<at::Tensor>& offsets, bool scale_grad_by_freq, int64_t mode, bool pruned,
    const std::optional<at::Tensor>& weights, const std::optional<at::Tensor>& mapping, bool include_last) {
  (void)scale_grad_by_freq; (void)mode;
  return quantized_embedding_byte_lookup(weight, indices, offsets, pruned, weights, mapping, include_last, false);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl("embedding_bag_prepack", TORCH_FN(quantized_embedding_prepack));
}
TORCH_LIBRARY_IMPL(quantized, PrivateUse1, m) {
  m.impl("embedding_bag_byte_prepack", TORCH_FN(quantized_embedding_byte_prepack));
  m.impl("embedding_bag_byte_prepack_with_rowwise_min_max", TORCH_FN(quantized_embedding_byte_prepack_minmax));
  m.impl("embedding_bag_byte_unpack", TORCH_FN(quantized_embedding_byte_unpack));
  m.impl("embedding_bag_byte_rowwise_offsets", TORCH_FN(quantized_embedding_byte_rowwise));
  m.impl("embedding_bag_byte", TORCH_FN(quantized_embedding_bag_byte));
  m.impl("embedding_byte", TORCH_FN(quantized_embedding_byte));
}
