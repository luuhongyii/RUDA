const char* arithmetic_dispatch_names[] = {
  "add.Tensor", "add_.Tensor", "sub.Tensor", "sub_.Tensor",
  "mul.Tensor", "mul_.Tensor", "neg", "neg_"
};
PyObject* arithmetic_python_bodies[8] = {};

void initialize_arithmetic_dispatch(const std::string& name, pybind11::object body) {
  for (size_t i = 0; i < 8; ++i) if (name == arithmetic_dispatch_names[i]) {
    TORCH_CHECK(!arithmetic_python_bodies[i] && PyCallable_Check(body.ptr()),
                "invalid or repeated arithmetic dispatcher initialization");
    arithmetic_python_bodies[i] = body.release().ptr();
    return;
  }
  TORCH_CHECK(false, "unknown arithmetic dispatcher: ", name);
}

bool native_arithmetic_tensor(const at::Tensor& value) {
  return value.device().type() == c10::DeviceType::PrivateUse1 && value.layout() == at::kStrided &&
    !value.is_neg() && !value.is_conj() &&
    (value.scalar_type() == at::kFloat || value.scalar_type() == at::kHalf ||
     value.scalar_type() == at::kBFloat16 || value.scalar_type() == at::kDouble);
}

template<int Op, bool Inplace>
at::Tensor native_arithmetic_dispatch(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha = 1) {
  if (native_arithmetic_tensor(a) && native_arithmetic_tensor(b) &&
      a.device() == b.device() && a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
      (!Inplace || (a.is_contiguous() && b.is_contiguous())) && !alpha.isSymbolic()) {
    if constexpr (Inplace) {
      at::assert_no_internal_overlap(a);
      at::assert_no_partial_overlap(a, b);
    }
    int64_t coefficient = 0;
    if constexpr (Op <= 1) {
      at::native::alpha_check(a.scalar_type(), alpha);
      std::vector<int64_t> bits;
      foreach_binary_scalar_bits(bits, Op == 1 ? -alpha : alpha, a.scalar_type());
      coefficient = bits[0];
    }
    auto out = Inplace ? a : at::empty(a.sizes(), a.options());
    scientific_call(485, {a, b, out}, {Op == 3 ? 2 : Op == 2 ? 1 : 0, 0, 0, coefficient}, {});
    return out;
  }
  if constexpr (!Inplace && Op <= 2) {
    return eager_tensor_arithmetic<Op>(a, b, alpha);
  }
  pybind11::gil_scoped_acquire gil;
  constexpr size_t index = Op * 2 + size_t(Inplace);
  TORCH_CHECK(arithmetic_python_bodies[index], "RUDA arithmetic dispatcher is not initialized");
  auto body = pybind11::reinterpret_borrow<pybind11::object>(arithmetic_python_bodies[index]);
  if constexpr (Op == 3) return body(a).template cast<at::Tensor>();
  else if constexpr (Op == 2) return body(a, b).template cast<at::Tensor>();
  else return body(a, b, pybind11::arg("alpha") = alpha).template cast<at::Tensor>();
}

template<int Op>
at::Tensor native_add_sub_dispatch(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  return native_arithmetic_dispatch<Op, false>(a, b, alpha);
}
template<int Op>
at::Tensor& native_add_sub_inplace_dispatch(at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  native_arithmetic_dispatch<Op, true>(a, b, alpha);
  return a;
}
at::Tensor native_mul_dispatch(const at::Tensor& a, const at::Tensor& b) {
  return native_arithmetic_dispatch<2, false>(a, b);
}
at::Tensor& native_mul_inplace_dispatch(at::Tensor& a, const at::Tensor& b) {
  native_arithmetic_dispatch<2, true>(a, b);
  return a;
}
at::Tensor native_neg_dispatch(const at::Tensor& a) {
  return native_arithmetic_dispatch<3, false>(a, a);
}
at::Tensor& native_neg_inplace_dispatch(at::Tensor& a) {
  native_arithmetic_dispatch<3, true>(a, a);
  return a;
}
