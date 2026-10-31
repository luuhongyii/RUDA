bool eager_real(const at::Tensor& value) {
  return eager_float(value) || eager_integral(value) ||
    (value.device().type()==c10::DeviceType::PrivateUse1 && value.layout()==at::kStrided &&
     value.scalar_type()==at::kDouble && !value.is_neg() && !value.is_conj());
}

template<typename Other>
at::Tensor eager_divide_fallback(const at::Tensor& a, const Other& b, std::optional<c10::string_view> mode) {
  pybind11::gil_scoped_acquire gil;
  auto rounding=mode.has_value() ? pybind11::cast(std::string(*mode)) : pybind11::none();
  return pybind11::module_::import("ruda_torch._division").attr("divide")(
      a,b,pybind11::arg("rounding_mode")=rounding).template cast<at::Tensor>();
}

int64_t eager_rounding(std::optional<c10::string_view> mode) {
  if (!mode.has_value()) return 0;
  if (*mode=="trunc") return 1;
  if (*mode=="floor") return 2;
  return -1;
}

std::optional<at::Tensor> eager_integer_quotient(const at::Tensor& a, const at::Scalar& b, int64_t op) {
  if (!eager_integer_scalar_ready || !eager_integral(a) || a.scalar_type()==at::kBool ||
      op<=0 || !b.isIntegral(true) || b.isSymbolic()) return std::nullopt;
  std::vector<int64_t> params{op,2};
  try { params.push_back(eager_integral_bits(b,a.scalar_type())); }
  catch (const c10::Error&) { return std::nullopt; }
  params.push_back(0);
  auto result=at::empty(a.sizes(),a.options());
  scientific_call(486,{a,a,result},params,{});
  return result;
}

at::Tensor eager_divide_tensor(const at::Tensor& a, const at::Tensor& b,
                              std::optional<c10::string_view> mode) {
  const auto op=eager_rounding(mode);
  if (!eager_arithmetic_ready || !eager_real(a) || op<0)
    return eager_divide_fallback(a,b,mode);
  if (b.device().is_cpu() && b.dim()==0 && !b.is_complex() && at::result_type(a,b)==a.scalar_type()) {
    auto result=eager_integer_quotient(a,b.item(),op);
    if (result.has_value()) return *result;
  }
  if (!eager_real(b) || a.device()!=b.device()) return eager_divide_fallback(a,b,mode);
  const auto dtype=at::result_type(a,b);
  if (dtype==at::kBool || (op==0 && c10::isIntegralType(dtype,true)))
    return eager_divide_fallback(a,b,mode);
  const auto shape=at::infer_size(a.sizes(),b.sizes());
  const auto left=a.to(dtype).expand(shape),right=b.to(dtype).expand(shape);
  auto result=at::empty(shape,a.options().dtype(dtype));
  scientific_call(486,{left,right,result},{op,0,0,0},{});
  return result;
}

at::Tensor eager_divide_scalar(const at::Tensor& a, const at::Scalar& b,
                              std::optional<c10::string_view> mode) {
  const auto op=eager_rounding(mode);
  if (op>=0 && eager_integral(a) && !b.isSymbolic() && at::result_type(a,b)==a.scalar_type()) {
    auto result=eager_integer_quotient(a,b,op);
    if (result.has_value()) return *result;
  }
  return eager_divide_fallback(a,b,mode);
}

template<int Op>
at::Tensor eager_compare(const at::Tensor& a, const at::Tensor& b) {
  if (!eager_arithmetic_ready || !eager_real(a) || !eager_real(b) || a.device()!=b.device()) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._predicates").attr("predicate")(
        a,b,pybind11::arg("op")=Op).template cast<at::Tensor>();
  }
  const auto dtype=at::result_type(a,b);
  const auto shape=at::infer_size(a.sizes(),b.sizes());
  const auto left=a.to(dtype).expand(shape),right=b.to(dtype).expand(shape);
  auto result=at::empty(shape,a.options().dtype(at::kBool));
  scientific_call(484,{left,right,result},{Op},{});
  return result;
}

at::Tensor eager_masked_fill(const at::Tensor& a, const at::Tensor& mask, const at::Scalar& value) {
  if (!eager_arithmetic_ready || !eager_real(a) || !eager_integral(mask) ||
      mask.scalar_type()!=at::kBool || mask.device()!=a.device() || value.isSymbolic()) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._conditional_select").attr("masked_fill")(a,mask,value).cast<at::Tensor>();
  }
  const auto shape=at::infer_size(a.sizes(),mask.sizes());
  auto source=a.expand(shape),condition=mask.expand(shape);
  uint64_t word=0;
  AT_DISPATCH_ALL_TYPES_AND3(at::kBool,at::kHalf,at::kBFloat16,a.scalar_type(),"ruda_eager_masked_fill",[&] {
    word=scalar_bits<scalar_t>(value);
  });
  auto result=at::empty(shape,a.options());
  scientific_call(483,{source,condition,result},{static_cast<int64_t>(word),0},{});
  return result;
}

at::Tensor eager_bitwise_not(const at::Tensor& a) {
  if (!eager_arithmetic_ready || !eager_integral(a) || a.scalar_type()!=at::kBool ||
      uint64_t(a.numel())>std::numeric_limits<uint32_t>::max()) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._bitwise").attr("elementwise")(
        a,pybind11::arg("op")=92).cast<at::Tensor>();
  }
  auto result=at::empty(a.sizes(),a.options());
  scientific_call(484,{a,a,result},{88},{});
  return result;
}

at::Tensor eager_exp(const at::Tensor& a) {
  if (!eager_arithmetic_ready || !eager_real(a) || eager_integral(a)) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._exponential").attr("unary")(
        a,pybind11::arg("op")=0).cast<at::Tensor>();
  }
  auto result=at::empty_like(a);
  scientific_call(489,{a,result},{0},{});
  return result;
}

at::Tensor eager_clone(const at::Tensor& a, std::optional<at::MemoryFormat> format) {
  if (!eager_arithmetic_ready || !eager_real(a)) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._ops").attr("clone")(
        a,pybind11::arg("memory_format")=format).cast<at::Tensor>();
  }
  auto result=at::empty_like(a,a.options(),format);
  copy_from(a,result,false);
  return result;
}

template<bool Maximum>
std::tuple<at::Tensor,at::Tensor> eager_extreme_dim(const at::Tensor& a, int64_t dim, bool keepdim) {
  if (!ordering_native || !eager_real(a) || !a.is_contiguous() || a.dim()==0 || a.numel()==0 ||
      uint64_t(a.numel())>std::numeric_limits<uint32_t>::max() || (dim!=-1 && dim!=a.dim()-1)) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._ordering").attr("extreme_dim")(
        a,dim,keepdim,pybind11::arg("maximum")=Maximum).template cast<std::tuple<at::Tensor,at::Tensor>>();
  }
  const auto width=a.size(-1),rows=a.numel()/width;
  auto source=a.reshape({rows,width});
  auto values=at::empty({rows},a.options());
  auto indices=at::empty({rows},a.options().dtype(at::kLong));
  ordering_call(0,{source,values,indices},{Maximum ? 4 : 5,2,0});
  auto shape=a.sizes().vec();
  if (keepdim) shape.back()=1;else shape.pop_back();
  return {values.reshape(shape),indices.reshape(shape)};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("div.Tensor",[](const at::Tensor& a,const at::Tensor& b){return eager_divide_tensor(a,b,std::nullopt);});
  m.impl("div.Tensor_mode",eager_divide_tensor);
  m.impl("div.Scalar",[](const at::Tensor& a,const at::Scalar& b){return eager_divide_scalar(a,b,std::nullopt);});
  m.impl("div.Scalar_mode",eager_divide_scalar);
  m.impl("floor_divide",[](const at::Tensor& a,const at::Tensor& b){return eager_divide_tensor(a,b,"floor");});
  m.impl("floor_divide.Scalar",[](const at::Tensor& a,const at::Scalar& b){return eager_divide_scalar(a,b,"floor");});
  m.impl("eq.Tensor",eager_compare<78>);
  m.impl("ne.Tensor",eager_compare<79>);
  m.impl("lt.Tensor",eager_compare<80>);
  m.impl("le.Tensor",eager_compare<81>);
  m.impl("gt.Tensor",eager_compare<82>);
  m.impl("ge.Tensor",eager_compare<83>);
  m.impl("masked_fill.Scalar",eager_masked_fill);
  m.impl("bitwise_not",eager_bitwise_not);
  m.impl("exp",eager_exp);
  m.impl("clone",eager_clone);
  m.impl("max.dim",eager_extreme_dim<true>);
  m.impl("min.dim",eager_extreme_dim<false>);
}
