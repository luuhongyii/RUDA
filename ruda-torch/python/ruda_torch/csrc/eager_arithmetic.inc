bool eager_arithmetic_ready = false;
bool eager_cat_ready = false;

int64_t eager_scalar_float_bits(const at::Scalar& value) {
  if (!value.isFloatingPoint()) return static_cast<int64_t>(scalar_bits<float>(value));
  const float converted=static_cast<float>(value.toDouble());
  uint32_t bits;std::memcpy(&bits,&converted,sizeof(bits));
  return static_cast<int64_t>(bits);
}

bool eager_float(const at::Tensor& value) {
  const auto dtype = value.scalar_type();
  return value.device().type() == c10::DeviceType::PrivateUse1 && value.layout() == at::kStrided &&
      !value.is_conj() && !value.is_neg() &&
      (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16);
}

bool eager_native_math() {
  pybind11::gil_scoped_acquire gil;
  return !pybind11::module_::import("ruda_torch.math_backend").attr("_nvidia")().cast<bool>();
}

template <bool Batched>
at::Tensor eager_matmul(const at::Tensor& a, const at::Tensor& b) {
  constexpr int64_t rank=Batched ? 3 : 2;
  if (!eager_arithmetic_ready || !eager_float(a) || !eager_float(b) ||
      a.device()!=b.device() || a.scalar_type()!=b.scalar_type() ||
      a.dim()!=rank || b.dim()!=rank || a.size(-1)!=b.size(-2) ||
      (Batched && a.size(0)!=b.size(0)) || !eager_native_math()) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._ops").attr(Batched ? "bmm" : "mm")(a,b).cast<at::Tensor>();
  }
  auto shape=a.sizes().vec();shape.back()=b.size(-1);
  auto result=at::empty(shape,a.options());
  execute(Batched ? 30 : 7,a,b,result,0.f);
  return result;
}

at::Tensor eager_addmm(const at::Tensor& bias, const at::Tensor& a, const at::Tensor& b,
                      const at::Scalar& beta, const at::Scalar& alpha) {
  if (!eager_arithmetic_ready || !eager_float(a) || !eager_float(b) || !eager_float(bias) ||
      a.device()!=b.device() || a.device()!=bias.device() ||
      a.scalar_type()!=b.scalar_type() || a.scalar_type()!=bias.scalar_type() ||
      a.dim()!=2 || b.dim()!=2 || a.size(1)!=b.size(0) || alpha.isComplex() || beta.isComplex() || !eager_native_math()) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._ops").attr("addmm")(
        bias,a,b,pybind11::arg("alpha")=alpha,pybind11::arg("beta")=beta).cast<at::Tensor>();
  }
  const std::vector<int64_t> shape{a.size(0),b.size(1)};
  auto expanded=bias.expand(shape);
  auto result=at::empty(shape,a.options());
  addmm(expanded,a,b,result,static_cast<float>(alpha.toDouble()),static_cast<float>(beta.toDouble()));
  return result;
}

at::Tensor eager_silu(const at::Tensor& a) {
  if (!eager_arithmetic_ready || !eager_float(a)) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._activation").attr("silu")(a).cast<at::Tensor>();
  }
  auto result=at::empty_like(a);
  scientific_call(284,{a,a,a,result},{0},{0.,0.,0.});
  return result;
}

at::Tensor eager_gelu(const at::Tensor& a, std::string_view approximate) {
  if (!eager_arithmetic_ready || !eager_float(a) || (approximate!="none" && approximate!="tanh")) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._activation").attr("gelu")(
        a,pybind11::arg("approximate")=std::string(approximate)).cast<at::Tensor>();
  }
  auto result=at::empty_like(a);
  scientific_call(282,{a,a,a,result},{approximate=="tanh" ? 1 : 0},{0.,0.,0.});
  return result;
}

template <int Op, typename Other>
at::Tensor eager_arithmetic_fallback(const at::Tensor& a, const Other& b, const at::Scalar& alpha) {
  pybind11::gil_scoped_acquire gil;
  auto function = pybind11::module_::import("ruda_torch._arithmetic").attr("arithmetic");
  return function(a, b, pybind11::arg("op")=Op, pybind11::arg("alpha")=alpha).template cast<at::Tensor>();
}

template <int Op>
at::Tensor eager_tensor_arithmetic(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha=1) {
  const bool scalar_a=a.device().is_cpu() && a.dim()==0 && !a.is_complex();
  const bool scalar_b=b.device().is_cpu() && b.dim()==0 && !b.is_complex();
  if (eager_arithmetic_ready && ((scalar_a && eager_float(b)) || (scalar_b && eager_float(a)))) {
    const auto& value=scalar_a ? b : a;
    const auto& scalar=scalar_a ? a : b;
    const auto dtype=at::result_type(a,b);
    if constexpr (Op==1) {
      if (scalar.scalar_type()==at::kBool) return eager_arithmetic_fallback<Op>(a,b,alpha);
    }
    if (dtype!=at::kFloat && dtype!=value.scalar_type()) return eager_arithmetic_fallback<Op>(a,b,alpha);
    std::vector<int64_t> params{Op==2 ? 1 : 0,scalar_a ? 2 : 0,scalar_b ? 2 : 0};
    if constexpr (Op<=1) {
      at::native::alpha_check(dtype,alpha);
      params.push_back(static_cast<int64_t>(scalar_bits<float>(Op==1 ? -alpha : alpha)));
    } else {params.push_back(0);}
    params.push_back(eager_scalar_float_bits(scalar.item()));
    auto result=at::empty(value.sizes(),value.options().dtype(dtype));
    scientific_call(485,{value,value,result},params,{});
    return result;
  }
  if (!eager_arithmetic_ready || !eager_float(a) || !eager_float(b) || a.device()!=b.device())
    return eager_arithmetic_fallback<Op>(a,b,alpha);
  const auto dtype=at::result_type(a,b);
  if (dtype!=at::kFloat && (a.scalar_type()!=dtype || b.scalar_type()!=dtype))
    return eager_arithmetic_fallback<Op>(a,b,alpha);
  std::vector<int64_t> params{Op==2 ? 1 : 0,0,0};
  if constexpr (Op<=1) {
    at::native::alpha_check(dtype,alpha);
    params.push_back(static_cast<int64_t>(scalar_bits<float>(Op==1 ? -alpha : alpha)));
  } else {params.push_back(0);}
  const auto shape=at::infer_size(a.sizes(),b.sizes());
  const auto left=a.sizes()==shape ? a : a.expand(shape);
  const auto right=b.sizes()==shape ? b : b.expand(shape);
  auto result=at::empty(shape,a.options().dtype(dtype));
  scientific_call(485,{left,right,result},params,{});
  return result;
}

template <int Op>
at::Tensor eager_scalar_arithmetic(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha=1) {
  if (!eager_arithmetic_ready || !eager_float(a) || b.isComplex() || (Op==1 && b.isBoolean()))
    return eager_arithmetic_fallback<Op>(a,b,alpha);
  std::vector<int64_t> params{Op==2 ? 1 : 0,0,2};
  if constexpr (Op<=1) {
    at::native::alpha_check(a.scalar_type(),alpha);
    params.push_back(static_cast<int64_t>(scalar_bits<float>(Op==1 ? -alpha : alpha)));
  } else {params.push_back(0);}
  params.push_back(eager_scalar_float_bits(b));
  auto result=at::empty(a.sizes(),a.options());
  scientific_call(485,{a,a,result},params,{});
  return result;
}

at::Tensor eager_neg(const at::Tensor& a) {
  if (!eager_arithmetic_ready || !eager_float(a))
    return eager_arithmetic_fallback<3>(a,a,1);
  auto result=at::empty(a.sizes(),a.options());
  scientific_call(485,{a,a,result},{2,0,0,0},{});
  return result;
}

at::Tensor eager_cat(const at::ITensorListRef& inputs, int64_t dim) {
  const std::vector<at::Tensor> tensors(inputs.begin(),inputs.end());
  bool eligible=tensors.size()==2;
  if (eligible) {
    const auto& a=tensors[0];const auto& b=tensors[1];
    eligible=eager_float(a) && eager_float(b) && a.device()==b.device() && a.scalar_type()==b.scalar_type() &&
        a.dim()>0 && a.dim()==b.dim() && dim>=-a.dim() && dim<a.dim() &&
        a.suggest_memory_format()==c10::MemoryFormat::Contiguous && b.suggest_memory_format()==c10::MemoryFormat::Contiguous;
    if (eligible) {
      dim=at::maybe_wrap_dim(dim,a.dim());
      for (int64_t axis=0;axis<a.dim();++axis)
        if (axis!=dim && a.size(axis)!=b.size(axis)) eligible=false;
    }
    if (eligible) {
      auto shape=a.sizes().vec();shape[dim]+=b.size(dim);
      auto result=at::empty(shape,a.options());
      if (eager_cat_ready) {
        execute(142,a,b,result,static_cast<float>(dim));
        return result;
      }
      auto left=result.as_strided(a.sizes(),result.strides(),0);
      auto right=result.as_strided(b.sizes(),result.strides(),a.size(dim)*result.stride(dim));
      if (a.numel()) copy_from(a,left,false);
      if (b.numel()) copy_from(b,right,false);
      return result;
    }
  }
  pybind11::gil_scoped_acquire gil;
  return pybind11::module_::import("ruda_torch._ops").attr("cat")(tensors,dim).cast<at::Tensor>();
}

at::Tensor eager_power(const at::Tensor& a, const at::Scalar& exponent) {
  auto fallback=[&]() {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._power").attr("power")(a,exponent).cast<at::Tensor>();
  };
  if (!eager_arithmetic_ready || !eager_float(a) || exponent.isComplex()) return fallback();
  int64_t special=0,bits=0;
  if (exponent.equal(0.0)) special=1;
  else if (exponent.equal(1.0)) special=2;
  else if (exponent.equal(0.5)) special=3;
  else if (exponent.equal(-0.5)) special=4;
  else if (exponent.equal(-1.0)) special=5;
  else {
    try {
      float value;
      if (a.scalar_type()==at::kHalf) value=static_cast<float>(exponent.to<at::Half>());
      else if (a.scalar_type()==at::kBFloat16) value=static_cast<float>(exponent.to<at::BFloat16>());
      else value=exponent.to<float>();
      bits=power_pack(value);
      if (value==2) special=6;else if (value==3) special=7;else if (value==-2) special=8;
    } catch (const c10::Error&) {return fallback();}
  }
  auto result=at::empty(a.sizes(),a.options());
  if (special==1) scientific_fill(result,1,0);
  else if (special==2) copy_from(a,result,false);
  else scientific_call(487,{a,a,result},{1,special,bits,0},{});
  return result;
}

at::Tensor eager_rsqrt(const at::Tensor& a) {
  if (!eager_arithmetic_ready || !eager_float(a)) {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._unary_basic").attr("unary")(a,pybind11::arg("op")=8).cast<at::Tensor>();
  }
  auto result=at::empty_like(a);
  scientific_call(487,{a,a,result},{1,4,0,0},{});
  return result;
}

at::Tensor eager_mean(const at::Tensor& a, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  auto fallback=[&]() {
    pybind11::gil_scoped_acquire gil;
    auto dimensions=dim.has_value() ? pybind11::cast(dim->vec()) : pybind11::none();
    return pybind11::module_::import("ruda_torch._additive_reduce").attr("mean_reduce")(
        a,dimensions,keepdim,pybind11::arg("dtype")=dtype).cast<at::Tensor>();
  };
  const auto output_type=dtype.value_or(a.scalar_type());
  if (!eager_arithmetic_ready || !eager_float(a) || a.numel()==0 ||
      (output_type!=a.scalar_type() && output_type!=at::kFloat)) return fallback();
  const auto rank=a.dim();
  std::vector<int64_t> flags(std::max<int64_t>(rank,1),0);
  if (!dim.has_value() || dim->empty()) std::fill(flags.begin(),flags.end(),1);
  else for (auto axis:*dim) {
    if (axis < -std::max<int64_t>(rank,1) || axis >= std::max<int64_t>(rank,1)) return fallback();
    axis=at::maybe_wrap_dim(axis,rank);
    if (flags[axis]) return fallback();
    flags[axis]=1;
  }
  std::vector<int64_t> shape;
  for (int64_t axis=0;axis<rank;++axis) {
    if (!flags[axis]) shape.push_back(a.size(axis));
    else if (keepdim) shape.push_back(1);
  }
  auto result=at::empty(shape,a.options().dtype(output_type));
  flags.insert(flags.begin(),1);
  scientific_call(480,{a,result},flags,{});
  return result;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> eager_layer_norm(const at::Tensor& a, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias, double eps) {
  using Result=std::tuple<at::Tensor,at::Tensor,at::Tensor>;
  auto fallback=[&]() {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._normalization").attr("layer_norm")(
        a,normalized_shape.vec(),weight,bias,eps).cast<Result>();
  };
  if (!eager_arithmetic_ready || !eager_float(a) || !a.is_contiguous() || a.numel()==0 ||
      normalized_shape.empty() || normalized_shape.size()>static_cast<size_t>(a.dim())) return fallback();
  int64_t width=1;
  for (size_t axis=0;axis<normalized_shape.size();++axis) {
    if (normalized_shape[axis]!=a.size(a.dim()-normalized_shape.size()+axis)) return fallback();
    width*=normalized_shape[axis];
  }
  for (const auto* parameter:{&weight,&bias}) {
    if (parameter->has_value() && parameter->value().defined()) {
      const auto& value=parameter->value();
      if (!eager_float(value) || !value.is_contiguous() || value.device()!=a.device() ||
          value.scalar_type()!=a.scalar_type() || value.sizes()!=normalized_shape) return fallback();
    }
  }
  if (training_native && normalized_shape.size()==1 && std::isfinite(eps) &&
      eps>=std::numeric_limits<float>::min() && eps<=std::numeric_limits<float>::max()) {
    auto [value,mean,inverse]=training_layer_forward(a,weight,bias,eps);
    auto stat_shape=a.sizes().vec();stat_shape.back()=1;
    return {value,mean.reshape(stat_shape),inverse.reshape(stat_shape)};
  }
  const auto rows=a.numel()/width;
  const auto x=a.reshape({rows,width});
  const auto options=a.options().dtype(at::kFloat);
  auto mean=at::empty({rows},options),variance=at::empty({rows},options),inverse=at::empty({rows},options);
  bool vectorized=width<=(1LL<<24) && width%4==0 && reinterpret_cast<uintptr_t>(a.data_ptr())%(4*a.element_size())==0;
  for (const auto* parameter:{&weight,&bias})
    if (parameter->has_value() && parameter->value().defined()) {
      const auto& value=parameter->value();
      vectorized=vectorized && reinterpret_cast<uintptr_t>(value.data_ptr())%(4*value.element_size())==0;
    }
  scientific_call(309,{x,mean,variance,inverse},{0,vectorized ? 1 : 0},{eps});
  const bool weighted=weight.has_value() && weight->defined(),biased=bias.has_value() && bias->defined();
  const auto w=weighted ? weight->to(at::kFloat).reshape({width}) : at::empty({0},options);
  const auto b=biased ? bias->to(at::kFloat).reshape({width}) : at::empty({0},options);
  auto result=at::empty({rows,width},a.options());
  scientific_call(310,{x,mean,inverse,w,b,result},{0,0,1,1,weighted ? 1 : 0,biased ? 1 : 0},{});
  auto stat_shape=a.sizes().vec();
  for (size_t axis=stat_shape.size()-normalized_shape.size();axis<stat_shape.size();++axis) stat_shape[axis]=1;
  return {result.reshape(a.sizes()),mean.reshape(stat_shape),inverse.reshape(stat_shape)};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("silu",eager_silu);
  m.impl("gelu",eager_gelu);
  m.impl("add.Scalar",[](const at::Tensor& a,const at::Scalar& b,const at::Scalar& alpha){return eager_scalar_arithmetic<0>(a,b,alpha);});
  m.impl("sub.Scalar",[](const at::Tensor& a,const at::Scalar& b,const at::Scalar& alpha){return eager_scalar_arithmetic<1>(a,b,alpha);});
  m.impl("mul.Scalar",[](const at::Tensor& a,const at::Scalar& b){return eager_scalar_arithmetic<2>(a,b);});
  m.impl("cat",eager_cat);
  m.impl("pow.Tensor_Scalar",eager_power);
  m.impl("rsqrt",eager_rsqrt);
  m.impl("mean.dim",eager_mean);
  m.impl("native_layer_norm",eager_layer_norm);
}
