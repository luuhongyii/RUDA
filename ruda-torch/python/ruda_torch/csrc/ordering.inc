using OrderingDispatch = int (*)(uint32_t,const Descriptor*,size_t,const int64_t*,size_t);
OrderingDispatch ordering_native = nullptr;

void ordering_call(uint32_t kind,const std::vector<at::Tensor>& tensors,const std::vector<int64_t>& integers) {
  TORCH_CHECK(ordering_native,"RUDA ordering API 1 unavailable; rebuild native libraries");
  TORCH_CHECK(kind<=2 && tensors.size()>=2 && tensors.size()<=3 && integers.size()<=(kind==0?3:1),
              "invalid ordering signature");
  const size_t inputs=kind==2?2:1;
  for(size_t i=0;i<tensors.size();++i) {
    const auto& t=tensors[i];
    validate(t.device());
    TORCH_CHECK(t.device()==tensors[0].device() && t.layout()==c10::Layout::Strided && t.is_contiguous(),
                "ordering requires contiguous tensors on the same device");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(),"ordering indexing exceeds U32 range");
    if(i>=inputs) {
      at::assert_no_internal_overlap(t);
      for(size_t j=0;j<i;++j)at::assert_no_overlap(t,tensors[j]);
    }
  }
  std::vector<Argument> arguments;arguments.reserve(tensors.size());
  for(const auto& tensor:tensors)arguments.emplace_back(tensor);
  std::vector<Descriptor> descriptors;descriptors.reserve(arguments.size());
  for(const auto& argument:arguments)descriptors.push_back(argument.desc);
  check(ordering_native(kind,descriptors.data(),descriptors.size(),integers.data(),integers.size()));
}
