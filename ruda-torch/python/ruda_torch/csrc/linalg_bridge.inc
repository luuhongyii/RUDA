using LinalgBridge = int (*)(uint32_t,const Descriptor*,size_t,const int64_t*,size_t,const double*,size_t);
LinalgBridge linalg_bridge_native = nullptr;
void linalg_bridge_call(uint32_t op,const std::vector<at::Tensor>& tensors,const std::vector<int64_t>& ints,const std::vector<double>& scalars) {
  TORCH_CHECK(linalg_bridge_native,"RUDA linalg API 1 unavailable; rebuild Rust and C++ libraries");
  TORCH_CHECK(!tensors.empty(),"linalg descriptors are required");
  std::vector<Argument> arguments;arguments.reserve(tensors.size());
  for(const auto& t:tensors) {validate(t.device());TORCH_CHECK(t.scalar_type()==at::kFloat||t.scalar_type()==at::kDouble||t.scalar_type()==at::kComplexFloat||t.scalar_type()==at::kComplexDouble||t.scalar_type()==at::kInt||(op==19&&(t.scalar_type()==at::kHalf||t.scalar_type()==at::kBFloat16||t.scalar_type()==at::kComplexHalf)),"RUDA linalg uses float32/float64, complex64/complex128 and int32 info/pivots; matrix_exp also accepts float16/bfloat16/complex32");arguments.emplace_back(t);}
  size_t begin=0;
  switch(op) {case 0:case 2:case 5:case 7:case 8:case 9:case 10:case 12:case 13:case 16:case 18:case 19:begin=1;break;case 1:case 11:case 17:begin=3;break;case 3:case 4:case 6:case 14:case 15:begin=2;break;case 255:begin=tensors.size();break;default:TORCH_CHECK(false,"unsupported linalg command");}
  for(size_t i=begin;i<tensors.size();++i) {at::assert_no_internal_overlap(tensors[i]);for(size_t j=0;j<i;++j) at::assert_no_overlap(tensors[i],tensors[j]);}
  std::vector<Descriptor> ds;ds.reserve(arguments.size());for(const auto& a:arguments)ds.push_back(a.desc);
  const int result=linalg_bridge_native(op,ds.data(),ds.size(),ints.data(),ints.size(),scalars.data(),scalars.size());
  if(op==255) {TORCH_CHECK_LINALG(result==0,error_native?error_native():"RUDA linalg status error");} else {check(result);}
}
