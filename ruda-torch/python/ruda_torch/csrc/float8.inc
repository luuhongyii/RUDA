using Float8Dispatch = int (*)(uint32_t,const Descriptor*,size_t,bool,bool,uint32_t);
Float8Dispatch float8_native=nullptr;
using Float8BlockDispatch = int (*)(uint32_t,const Descriptor*,size_t,uint32_t,uint32_t,uint32_t,uint32_t,bool,uint32_t);
Float8BlockDispatch float8_block_native=nullptr;
using Float8ShardDispatch = int (*)(uint32_t,const Descriptor*,size_t,const int64_t*,size_t);
Float8ShardDispatch float8_shard_native=nullptr;

void float8_shard_dispatch(uint32_t op,const std::vector<at::Tensor>& tensors,const std::vector<int64_t>& integers) {
  TORCH_CHECK(float8_shard_native,"RUDA sharded block FP8 API unavailable; rebuild native libraries");
  TORCH_CHECK(op<=1 && tensors.size()==(op==0?2:4) && integers.size()==(op==0?6:8),"invalid sharded FP8 dispatch");
  std::vector<Argument> args;args.reserve(tensors.size());
  for(const auto& t:tensors) {
    validate(t.device());
    TORCH_CHECK(t.device()==tensors[0].device() && t.is_contiguous(),"sharded FP8 operands require contiguous same-device storage");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(),"sharded FP8 operand exceeds U32 indexing");
    args.emplace_back(t);
  }
  for(size_t i=op==0?1:2;i<tensors.size();++i) {
    at::assert_no_internal_overlap(tensors[i]);
    for(size_t j=0;j<i;++j)at::assert_no_overlap(tensors[i],tensors[j]);
  }
  std::vector<Descriptor> ds;ds.reserve(args.size());
  for(const auto& arg:args)ds.push_back(arg.desc);
  check(float8_shard_native(op,ds.data(),ds.size(),integers.data(),integers.size()));
}

void float8_block_dispatch(uint32_t op,const std::vector<at::Tensor>& tensors,uint32_t rows_a,
    uint32_t rows_w,uint32_t block_k,uint32_t margin,bool power_of_two,uint32_t strategy) {
  TORCH_CHECK(float8_block_native,"RUDA block FP8 API unavailable; rebuild native libraries");
  TORCH_CHECK(op<=1 && strategy<=2 && tensors.size()==(op==0?5:3),"invalid block FP8 dispatch");
  std::vector<Argument> args;args.reserve(tensors.size());
  for(const auto& t:tensors) {
    validate(t.device());
    TORCH_CHECK(t.device()==tensors[0].device() && t.is_contiguous(),"block FP8 operands need contiguous same-device storage");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(),"block FP8 operand exceeds U32 indexing");
    args.emplace_back(t);
  }
  size_t first_output=op==0?4:1;
  for(size_t i=first_output;i<tensors.size();++i) {
    at::assert_no_internal_overlap(tensors[i]);
    for(size_t j=0;j<i;++j) at::assert_no_overlap(tensors[i],tensors[j]);
  }
  std::vector<Descriptor> ds;ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  check(float8_block_native(op,ds.data(),ds.size(),rows_a,rows_w,block_k,margin,power_of_two,strategy));
}

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
