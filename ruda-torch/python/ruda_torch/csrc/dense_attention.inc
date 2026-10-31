using DenseAttention = int (*)(uint32_t,const Descriptor*,size_t,uint32_t,bool,float,float,uint64_t,uint64_t,bool);
DenseAttention dense_attention_native = nullptr;
using IndexedDenseAttention = int (*)(uint32_t,const Descriptor*,size_t,uint32_t,bool,float,float,uint64_t,uint64_t,bool,int64_t,int64_t);
IndexedDenseAttention indexed_dense_attention_native = nullptr;
using AttentionDropout = int (*)(const Descriptor*,const int64_t*,float,uint64_t,uint64_t);
AttentionDropout attention_dropout_native = nullptr;

void attention_dropout_call(const std::vector<at::Tensor>& tensors,const std::vector<int64_t>& integers,
    double dropout,uint64_t seed,uint64_t counter) {
  TORCH_CHECK(attention_dropout_native && tensors.size()==2 && integers.size()==7,"invalid attention dropout command");
  training_tensor(tensors[0]);training_tensor(tensors[1]);
  TORCH_CHECK(tensors[0].device()==tensors[1].device() && tensors[0].scalar_type()==tensors[1].scalar_type(),
    "attention dropout dtype/device mismatch");
  TORCH_CHECK(tensors[0].numel()==0 && tensors[1].dim()==4 && dropout>=0 && dropout<=1,"invalid attention dropout tile");
  at::assert_no_internal_overlap(tensors[1]);at::assert_no_overlap(tensors[0],tensors[1]);
  Argument anchor(tensors[0]),output(tensors[1]);Descriptor descriptors[]={anchor.desc,output.desc};
  check(attention_dropout_native(descriptors,integers.data(),float(dropout),seed,counter));
}

void dense_attention_dispatch(uint32_t op,const std::vector<at::Tensor>& tensors,
    uint32_t mask_kind,const std::vector<int64_t>& score_shape,bool causal,
    double scale,double p,uint64_t seed,uint64_t counter,bool need_mask,bool indexed,int64_t global_heads,int64_t head_start) {
  TORCH_CHECK(indexed?bool(indexed_dense_attention_native):bool(dense_attention_native),
    "RUDA dense attention API unavailable; rebuild Rust and C++ libraries");
  TORCH_CHECK(op<=1 && tensors.size()==(op==0?6:10) && mask_kind<=2,"invalid dense attention command");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(float(scale)) && p>=0 && p<=1,"invalid dense attention scale/dropout");
  TORCH_CHECK(!causal || mask_kind==0,"causal and explicit masks are mutually exclusive");
  for(size_t i=0;i<tensors.size();++i) {
    const auto& t=tensors[i];
    if(i<3 || (op==1 && i>=6 && i<9)) {
      validate(t.device());
      TORCH_CHECK(!t.is_conj() && !t.is_neg() && dtype_code(t.scalar_type())<=2,"attention requires resolved FP32/FP16/BF16 operands");
    } else training_tensor(t);
  }
  TORCH_CHECK(tensors[3].scalar_type()==at::kFloat,"dense mask storage must be FP32");
  // Metadata-only expansion; its strides address the compact mask and the
  // compact dMask directly. No expanded mask or score matrix is allocated.
  auto mask = mask_kind ? tensors[3].expand(score_shape) : tensors[3];
  const size_t begin=op==0?4:6;
  for(size_t i=begin;i<tensors.size();++i) {
    at::assert_no_internal_overlap(tensors[i]);
    for(size_t j=0;j<i;++j) at::assert_no_overlap(tensors[i],tensors[j]);
  }
  if(op==1 && tensors[0].numel()!=0)
    at::globalContext().alertNotDeterministic("ruda::dense_attention_backward");
  auto native_tensors=tensors;
  if(op==1) {
    for(size_t i=0;i<3;++i)native_tensors[i]=tensors[i].contiguous();
    for(size_t i=6;i<9;++i)if(!tensors[i].is_contiguous())native_tensors[i]=at::empty(tensors[i].sizes(),tensors[i].options());
  }
  std::vector<Argument> args; args.reserve(tensors.size());
  for(size_t i=0;i<tensors.size();++i)args.emplace_back(i==3?mask:native_tensors[i]);
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  if(indexed)check(indexed_dense_attention_native(op,ds.data(),ds.size(),mask_kind,causal,float(scale),float(p),seed,counter,need_mask,global_heads,head_start));
  else check(dense_attention_native(op,ds.data(),ds.size(),mask_kind,causal,float(scale),float(p),seed,counter,need_mask));
  if(op==1)for(size_t i=6;i<9;++i)if(!tensors[i].is_contiguous())tensors[i].copy_(native_tensors[i]);
}

void dense_attention_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    uint32_t mask_kind,const std::vector<int64_t>& score_shape,bool causal,
    double scale,double p,uint64_t seed,uint64_t counter,bool need_mask) {
  dense_attention_dispatch(op,tensors,mask_kind,score_shape,causal,scale,p,seed,counter,need_mask,false,0,0);
}

at::Tensor dense_attention_inference(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& mask,const std::vector<int64_t>& score_shape,uint32_t mask_kind,bool causal,double scale) {
  TORCH_CHECK(!c10::GradMode::is_enabled(),"dense inference requires disabled autograd");
  TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4,"dense inference requires canonical rank-four operands");
  auto output=at::empty({q.size(0),q.size(1),q.size(2),v.size(3)},q.options());
  auto stats=at::empty({q.size(0)*q.size(1)*q.size(2),2},q.options().dtype(at::kFloat));
  dense_attention_call(0,{q,k,v,mask,output,stats},mask_kind,score_shape,causal,scale,0.,0,0,false);
  return output;
}

void indexed_dense_attention_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    uint32_t mask_kind,const std::vector<int64_t>& score_shape,bool causal,
    double scale,double p,uint64_t seed,uint64_t counter,bool need_mask,int64_t global_heads,int64_t head_start) {
  dense_attention_dispatch(op,tensors,mask_kind,score_shape,causal,scale,p,seed,counter,need_mask,true,global_heads,head_start);
}

std::atomic<bool> dense_dispatch_ready{false}, dense_flash_ready{false};
void initialize_dense_dispatch(bool flash) {
  if(!dense_dispatch_ready.load(std::memory_order_acquire)) {
    pybind11::gil_scoped_acquire gil;
    if(!dense_dispatch_ready.load(std::memory_order_relaxed)) {
      pybind11::module_::import("ruda_torch.attention").attr("_initialize_dense")();
      dense_dispatch_ready.store(true,std::memory_order_release);
    }
  }
  if(flash && !dense_flash_ready.load(std::memory_order_acquire)) {
    pybind11::gil_scoped_acquire gil;
    if(!dense_flash_ready.load(std::memory_order_relaxed)) {
      pybind11::module_::import("ruda_torch._dense_ops").attr("_columns")(true);
      dense_flash_ready.store(true,std::memory_order_release);
    }
  }
}

std::vector<int64_t> dense_score_shape(c10::SymIntArrayRef shape) {
  std::vector<int64_t> result;result.reserve(shape.size());
  for(const auto& dim:shape) result.push_back(dim.expect_int());
  return result;
}

std::pair<uint64_t,uint64_t> dense_rng_words(const at::Tensor& state) {
  auto original = [&] {
    pybind11::gil_scoped_acquire gil;
    return pybind11::module_::import("ruda_torch._dense_ops").attr("rng_words")(state)
        .cast<std::pair<uint64_t,uint64_t>>();
  };
  if(!state.device().is_cpu() || state.scalar_type()!=at::kLong || state.dim()!=1 || state.size(0)!=4
      || state.is_neg() || state.is_conj())
    return original();
  const auto* data=state.const_data_ptr<int64_t>();
  uint64_t words[4];
  for(int64_t i=0;i<4;++i) {
    const auto word=data[i*state.stride(0)];
    if(word<0 || uint64_t(word)>0xffffffffULL) return original();
    words[i]=uint64_t(word);
  }
  return {words[0]|(words[1]<<32),words[2]|(words[3]<<32)};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> dense_attention_forward_op(
    const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& mask,
    c10::SymIntArrayRef score_shape,c10::SymInt mask_kind,bool causal,double scale,double p,bool flash) {
  initialize_dense_dispatch(flash);
  TORCH_CHECK(q.dim()>=3 && v.dim()>0,"invalid dense attention dimensions");
  auto shape=q.sizes().vec();shape.back()=v.size(-1);
  auto out=at::empty(shape,q.options());
  auto stats=at::empty({q.size(0)*q.size(1)*q.size(2),flash?130:2},q.options().dtype(at::kFloat));
  auto score=dense_score_shape(score_shape);
  at::Tensor state;
  if(p>0 && p<1) {
    pybind11::gil_scoped_acquire gil;
    auto count=pybind11::module_::import("math").attr("prod")(score);
    state=pybind11::module_::import("ruda_torch._dense_ops").attr("_rng_state")(count,p).cast<at::Tensor>();
  } else state=at::zeros({4},at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto [seed,counter]=dense_rng_words(state);
  TORCH_CHECK(mask_kind.expect_int()>=0 && mask_kind.expect_int()<=2,"invalid dense attention command");
  dense_attention_call(0,{q,k,v,mask,out,stats},mask_kind.expect_int(),score,causal,scale,p,seed,counter,false);
  return {out,stats,state};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> dense_attention_backward_op(
    const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& mask,
    const at::Tensor& grad,const at::Tensor& stats,const at::Tensor& rng,c10::SymIntArrayRef score_shape,
    c10::SymInt mask_kind,bool causal,double scale,double p,bool need_mask) {
  initialize_dense_dispatch(false);
  auto dq=at::empty_like(q),dk=at::empty_like(k),dv=at::empty_like(v);
  auto dm=need_mask?at::empty_like(mask):at::empty({1},q.options().dtype(at::kFloat));
  auto [seed,counter]=dense_rng_words(rng);
  TORCH_CHECK(mask_kind.expect_int()>=0 && mask_kind.expect_int()<=2,"invalid dense attention command");
  dense_attention_call(1,{q,k,v,mask,grad.contiguous(),stats,dq,dk,dv,dm},mask_kind.expect_int(),
      dense_score_shape(score_shape),causal,scale,p,seed,counter,need_mask);
  return {dq,dk,dv,dm};
}
