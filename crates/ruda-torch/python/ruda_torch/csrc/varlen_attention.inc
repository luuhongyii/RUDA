using VarlenAttention = int (*)(uint32_t,const Descriptor*,size_t,bool,bool,int64_t,int64_t,float,float,float,uint64_t,uint64_t);
VarlenAttention varlen_attention_native = nullptr;

void varlen_attention_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    bool causal,bool lower_right,int64_t left,int64_t right,double scale,double p,double softcap,
    uint64_t seed,uint64_t counter) {
  TORCH_CHECK(varlen_attention_native,"RUDA varlen attention API 1 unavailable; rebuild native components");
  TORCH_CHECK(op<=1 && tensors.size()==(op==0?6:9),"invalid varlen command");
  TORCH_CHECK(left>=-1 && right>=-1 && std::isfinite(scale) && std::isfinite(float(scale)) &&
              p>=0 && p<=1 && std::isfinite(softcap) && std::isfinite(float(softcap)) && softcap>=0,
              "invalid varlen attention options");
  for(size_t i=0;i<tensors.size();++i) {
    const auto& t=tensors[i];
    TORCH_CHECK(t.device()==tensors[0].device(),"varlen tensors must share a device");
    if(i==3) {
      validate(t.device());
      TORCH_CHECK(t.is_contiguous() && t.scalar_type()==at::kLong && t.dim()==2 && t.size(1)==6,
                  "varlen schedule must be contiguous int64 [documents,6]");
    } else training_tensor(t);
  }
  std::vector<Argument> args; args.reserve(tensors.size());
  for(const auto& t:tensors) args.emplace_back(t);
  const size_t begin=op==0?4:6;
  for(size_t i=begin;i<tensors.size();++i) {
    at::assert_no_internal_overlap(tensors[i]);
    for(size_t j=0;j<i;++j) at::assert_no_overlap(tensors[i],tensors[j]);
  }
  if(op==1 && tensors[0].numel()!=0)
    at::globalContext().alertNotDeterministic("ruda::varlen_attention_backward");
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  check(varlen_attention_native(op,ds.data(),ds.size(),causal,lower_right,left,right,
        float(scale),float(p),float(softcap),seed,counter));
}
