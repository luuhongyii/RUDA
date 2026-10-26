PyObject* matmul_backend_context = nullptr;
PyObject* matmul_python_bodies[3] = {nullptr, nullptr, nullptr};

void initialize_matmul_dispatch(pybind11::object context, pybind11::object mm,
                                pybind11::object bmm, pybind11::object addmm_body) {
  TORCH_CHECK(!matmul_backend_context && PyContextVar_CheckExact(context.ptr()) &&
              PyCallable_Check(mm.ptr()) && PyCallable_Check(bmm.ptr()) &&
              PyCallable_Check(addmm_body.ptr()), "invalid or repeated matmul dispatcher initialization");
  matmul_backend_context = context.release().ptr();
  matmul_python_bodies[0] = mm.release().ptr();
  matmul_python_bodies[1] = bmm.release().ptr();
  matmul_python_bodies[2] = addmm_body.release().ptr();
}

bool native_matmul_context() {
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(matmul_backend_context, "RUDA matmul dispatcher is not initialized");
  PyObject* selected = nullptr;
  if (PyContextVar_Get(matmul_backend_context, nullptr, &selected) < 0) throw pybind11::error_already_set();
  auto backend = pybind11::reinterpret_steal<pybind11::object>(selected);
  return selected && PyUnicode_Check(selected) && PyUnicode_CompareWithASCIIString(selected, "ruda") == 0;
}

bool native_matmul_operands(const at::Tensor& a, const at::Tensor& b, int64_t rank) {
  return a.dim() == rank && b.dim() == rank && a.device().type() == c10::DeviceType::PrivateUse1 &&
    a.device() == b.device() && a.scalar_type() == b.scalar_type() &&
    (a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16) &&
    a.layout() == at::kStrided && b.layout() == at::kStrided && a.size(rank - 1) == b.size(rank - 2) &&
    (rank == 2 || a.size(0) == b.size(0));
}

at::Tensor native_mm_dispatch(const at::Tensor& a, const at::Tensor& b) {
  if (native_matmul_operands(a, b, 2) && native_matmul_context()) {
    auto out = at::empty({a.size(0), b.size(1)}, a.options());
    execute(7, a, b, out, 0.0f);
    return out;
  }
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(matmul_python_bodies[0], "RUDA matmul dispatcher is not initialized");
  return pybind11::reinterpret_borrow<pybind11::object>(matmul_python_bodies[0])(a, b).cast<at::Tensor>();
}

at::Tensor native_bmm_dispatch(const at::Tensor& a, const at::Tensor& b) {
  if (native_matmul_operands(a, b, 3) && native_matmul_context()) {
    auto out = at::empty({a.size(0), a.size(1), b.size(2)}, a.options());
    execute(30, a, b, out, 0.0f);
    return out;
  }
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(matmul_python_bodies[1], "RUDA matmul dispatcher is not initialized");
  return pybind11::reinterpret_borrow<pybind11::object>(matmul_python_bodies[1])(a, b).cast<at::Tensor>();
}

at::Tensor native_addmm_dispatch(const at::Tensor& bias, const at::Tensor& a, const at::Tensor& b,
                                 const at::Scalar& beta, const at::Scalar& alpha) {
  if (native_matmul_operands(a, b, 2) && bias.device() == a.device() &&
      bias.scalar_type() == a.scalar_type() && bias.layout() == at::kStrided &&
      !alpha.isComplex() && !beta.isComplex() && !alpha.isSymbolic() && !beta.isSymbolic() && native_matmul_context()) {
    auto expanded = bias.expand({a.size(0), b.size(1)});
    auto out = at::empty({a.size(0), b.size(1)}, a.options());
    addmm(expanded, a, b, out, static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble()));
    return out;
  }
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(matmul_python_bodies[2], "RUDA matmul dispatcher is not initialized");
  return pybind11::reinterpret_borrow<pybind11::object>(matmul_python_bodies[2])(
    bias, a, b, pybind11::arg("beta") = beta, pybind11::arg("alpha") = alpha).cast<at::Tensor>();
}
