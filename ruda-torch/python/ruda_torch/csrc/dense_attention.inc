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
  for(const auto& t:tensors) training_tensor(t);
  TORCH_CHECK(tensors[3].scalar_type()==at::kFloat,"dense mask storage must be FP32");
  // Metadata-only expansion; its strides address the compact mask and the
  // compact dMask directly. No expanded mask or score matrix is allocated.
  auto mask = mask_kind ? tensors[3].expand(score_shape) : tensors[3];
  std::vector<Argument> args; args.reserve(tensors.size());
  for(size_t i=0;i<tensors.size();++i) args.emplace_back(i==3?mask:tensors[i]);
  const size_t begin=op==0?4:6;
  for(size_t i=begin;i<tensors.size();++i) {
    at::assert_no_internal_overlap(tensors[i]);
    for(size_t j=0;j<i;++j) at::assert_no_overlap(tensors[i],tensors[j]);
  }
  if(op==1 && tensors[0].numel()!=0)
    at::globalContext().alertNotDeterministic("ruda::dense_attention_backward");
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  if(indexed)check(indexed_dense_attention_native(op,ds.data(),ds.size(),mask_kind,causal,float(scale),float(p),seed,counter,need_mask,global_heads,head_start));
  else check(dense_attention_native(op,ds.data(),ds.size(),mask_kind,causal,float(scale),float(p),seed,counter,need_mask));
}

void dense_attention_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    uint32_t mask_kind,const std::vector<int64_t>& score_shape,bool causal,
    double scale,double p,uint64_t seed,uint64_t counter,bool need_mask) {
  dense_attention_dispatch(op,tensors,mask_kind,score_shape,causal,scale,p,seed,counter,need_mask,false,0,0);
}

void indexed_dense_attention_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    uint32_t mask_kind,const std::vector<int64_t>& score_shape,bool causal,
    double scale,double p,uint64_t seed,uint64_t counter,bool need_mask,int64_t global_heads,int64_t head_start) {
  dense_attention_dispatch(op,tensors,mask_kind,score_shape,causal,scale,p,seed,counter,need_mask,true,global_heads,head_start);
}
