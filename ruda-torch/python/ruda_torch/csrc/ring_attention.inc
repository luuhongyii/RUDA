using RingAttention = int (*)(uint32_t,const Descriptor*,size_t,size_t,uint32_t,bool,int64_t,int64_t,
                              float,float,float,uint64_t,uint64_t);
RingAttention ring_attention_native = nullptr;

void ring_attention_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    int64_t batch,uint32_t global_keys,bool causal,int64_t left,int64_t right,
    double scale,double p,double softcap,uint64_t seed,uint64_t counter) {
  TORCH_CHECK(ring_attention_native,"RUDA ring attention API 1 unavailable; rebuild native components");
  TORCH_CHECK(op<=2 && tensors.size()==(op==0?7:op==1?11:3) && batch>0,"invalid ring attention command");
  TORCH_CHECK(left>=-1 && right>=-1 && std::isfinite(scale) && std::isfinite(float(scale)) &&
              p>=0 && p<=1 && std::isfinite(softcap) && std::isfinite(float(softcap)) && softcap>=0,
              "invalid ring attention policy");
  std::vector<Argument> args; args.reserve(tensors.size());
  for(size_t i=0;i<tensors.size();++i) {
    const auto& t=tensors[i];
    TORCH_CHECK(t.device()==tensors[0].device(),"ring attention tensors must share a device");
    if(op!=2 && (i==3 || i==4)) {
      validate(t.device());
      TORCH_CHECK(t.is_contiguous() && t.scalar_type()==at::kLong && t.dim()==2 &&
                  (t.size(1)==3 || (i==3 && t.size(1)==4)),
                  "ring metadata requires int64 [tokens,3], or indexed query metadata [tokens,4]");
    } else training_tensor(t);
    args.emplace_back(t);
  }
  const size_t begin=op==0?5:op==1?8:0;
  for(size_t i=begin;i<tensors.size();++i) {
    at::assert_no_internal_overlap(tensors[i]);
    for(size_t j=0;j<i;++j) at::assert_no_overlap(tensors[i],tensors[j]);
  }
  if(op==1 && tensors[0].numel()!=0)
    at::globalContext().alertNotDeterministic("ruda::ring_attention_backward");
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  check(ring_attention_native(op,ds.data(),ds.size(),size_t(batch),global_keys,causal,left,right,
                              float(scale),float(p),float(softcap),seed,counter));
}
