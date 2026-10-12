using PackedDispatch = int (*)(uint32_t,const Descriptor*,size_t,const uint32_t*,size_t);
PackedDispatch packed_native=nullptr;
using QuantizedPaged = int (*)(const void*,const Descriptor*,size_t,const uint32_t*,size_t,float,bool);
QuantizedPaged quantized_paged_native=nullptr;

void packed_dispatch(uint32_t op,const std::vector<at::Tensor>& tensors,const std::vector<uint32_t>& layout) {
  TORCH_CHECK(packed_native,"RUDA packed quantization API 1 unavailable; rebuild both native libraries");
  TORCH_CHECK(tensors.size()==6 && layout.size()==4,"invalid packed ABI arguments");
  std::vector<Argument> args; args.reserve(tensors.size());
  for(const auto& t:tensors) {
    TORCH_CHECK(t.device()==tensors[0].device() && t.is_contiguous(),"packed operands require contiguous same-device storage");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(),"packed operand exceeds U32 indexing");
    args.emplace_back(t);
  }
  for(size_t i=0;i<5;++i) at::assert_no_overlap(tensors[5],tensors[i]);
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for(const auto& arg:args) ds.push_back(arg.desc);
  check(packed_native(op,ds.data(),ds.size(),layout.data(),layout.size()));
}
