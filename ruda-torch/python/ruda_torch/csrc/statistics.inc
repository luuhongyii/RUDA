using StatisticsDispatch = int (*)(const Descriptor*,size_t);
StatisticsDispatch statistics_native = nullptr;

void statistics_call(const at::Tensor& input,const at::Tensor& mean,const at::Tensor& m2) {
  TORCH_CHECK(statistics_native,"RUDA statistics API 1 unavailable; rebuild native libraries");
  TORCH_CHECK(input.dim()==2 && input.size(1)>0,"moments require nonempty packed rows");
  TORCH_CHECK(input.scalar_type()==at::kFloat || input.scalar_type()==at::kHalf ||
              input.scalar_type()==at::kBFloat16 || input.scalar_type()==at::kDouble,
              "moments require real floating input");
  const auto dtype=input.scalar_type()==at::kDouble?at::kDouble:at::kFloat;
  for(const auto& t:{input,mean,m2}) {
    validate(t.device());
    TORCH_CHECK(t.device()==input.device() && t.layout()==c10::Layout::Strided && t.is_contiguous(),
                "moments require contiguous tensors on the same device");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(),"moments indexing exceeds U32 range");
  }
  for(const auto& t:{mean,m2}) {
    TORCH_CHECK(t.dim()==1 && t.size(0)==input.size(0) && t.scalar_type()==dtype,"invalid moments output");
    at::assert_no_internal_overlap(t);
    at::assert_no_overlap(t,input);
  }
  at::assert_no_overlap(mean,m2);
  Argument a(input),b(mean),c(m2);
  Descriptor descriptors[]={a.desc,b.desc,c.desc};
  check(statistics_native(descriptors,3));
}
