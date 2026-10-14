using Float8Dispatch = int (*)(uint32_t,const Descriptor*,size_t,bool,bool,uint32_t);
Float8Dispatch float8_native=nullptr;

void float8_dispatch(uint32_t op,const std::vector<at::Tensor>& tensors,bool ta,bool tb,uint32_t strategy) {
  TORCH_CHECK(float8_native,"RUDA FP8 API 1 unavailable; rebuild native libraries");
  TORCH_CHECK(op<=1 && strategy<=2 && tensors.size()==(op==0?5:3),"invalid FP8 dispatch");
  std::vector<Argument> args;args.reserve(tensors.size());
  for(const auto& t:tensors) {
    validate(t.device());
    TORCH_CHECK(t.device()==tensors[0].device() && t.is_contiguous(),"FP8 operands need contiguous same-device storage");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(),"FP8 operand exceeds U32 indexing");
    args.emplace_back(t);
  }
  const auto& out=tensors.back();at::assert_no_internal_overlap(out);
  for(size_t i=0;i+1<tensors.size();++i) at::assert_no_overlap(out,tensors[i]);
  std::vector<Descriptor> ds;ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  check(float8_native(op,ds.data(),ds.size(),ta,tb,strategy));
}
