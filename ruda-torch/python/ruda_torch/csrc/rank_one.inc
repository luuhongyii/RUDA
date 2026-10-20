template <typename T>
void rank_one_scalar_bits(const T& value, int64_t* bits) {
  if constexpr (c10::is_complex<T>::value) {
    const auto real = value.real(), imag = value.imag();
    std::memcpy(bits, &real, sizeof(real));
    std::memcpy(bits + 1, &imag, sizeof(imag));
  } else {
    std::memcpy(bits, &value, sizeof(value));
  }
}

void addr_kernel_ruda(at::TensorIterator& iter, const at::Scalar& beta, const at::Scalar& alpha) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool, iter.dtype(), "addr_ruda", [&] {
    const auto beta_value = beta.to<scalar_t>(), alpha_value = alpha.to<scalar_t>();
    std::vector<int64_t> bits(4, 0);
    rank_one_scalar_bits(alpha_value, bits.data());
    rank_one_scalar_bits(beta_value, bits.data() + 2);
    auto out = iter.output();
    if (!out.numel()) return;
    auto result = out.is_conj() || out.is_neg() ? at::empty(out.sizes(), out.options()) : out;
    std::vector<at::Tensor> tensors;
    tensors.reserve(4);
    for (int64_t index = 0; index < 3; ++index) {
      tensors.push_back(scientific_resolve(iter.input(index)).to(iter.dtype()).expand(out.sizes()));
    }
    tensors.push_back(result);
    scientific_call(494, tensors, bits, {});
    if (!result.is_same(out)) copy_from(result, out, false);
  });
  iter.cast_outputs();
}
