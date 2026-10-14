using SortingDispatch=int (*)(const Descriptor*,size_t,const int64_t*,size_t);
SortingDispatch sorting_native=nullptr;

std::tuple<at::Tensor,at::Tensor> sorting_result(const at::Tensor& self,int64_t mode,int64_t dim,
    int64_t k,bool descending,bool ordered,bool stable) {
  TORCH_CHECK(sorting_native,"RUDA sorting API 1 unavailable; rebuild native libraries");
  validate(self.device());
  TORCH_CHECK(self.layout()==c10::Layout::Strided && self.dim()<=32 && dtype_code(self.scalar_type())<=9,
              "RUDA sorting requires a real strided tensor of rank <=32");
  TORCH_CHECK(mode!=0 || self.scalar_type()!=at::kBool,"topk does not support bool");
  dim=at::maybe_wrap_dim(dim,self.dim(),true);
  const int64_t length=self.dim() ? self.size(dim) : 1;
  if(mode!=0)k=length;
  TORCH_CHECK(k>=0 && k<=length,"selected index k out of range");
  TORCH_CHECK(uint64_t(self.numel())<=std::numeric_limits<uint32_t>::max(),"sort indexing exceeds U32 range");
  auto shape=self.sizes().vec();
  if(mode==0 && !shape.empty())shape[dim]=k;
  const auto strides=mode==0 ? std::vector<int64_t>{} :
      (self.is_non_overlapping_and_dense() ? self.strides().vec() : at::infer_dense_strides(self.sizes(),self.strides()));
  auto allocate_result=[&](c10::ScalarType dtype) {
    return mode==0 ? at::empty(shape,self.options().dtype(dtype)) :
                    at::empty_strided(shape,strides,self.options().dtype(dtype));
  };
  at::Tensor values;
  if(mode!=2)values=allocate_result(self.scalar_type());
  auto indices=allocate_result(at::kLong);
  std::vector<at::Tensor> tensors{self.resolve_conj().resolve_neg().contiguous()};
  if(mode!=2)tensors.push_back(values);
  tensors.push_back(indices);
  std::vector<Argument> arguments;arguments.reserve(tensors.size());
  for(const auto& tensor:tensors)arguments.emplace_back(tensor);
  std::vector<Descriptor> descriptors;descriptors.reserve(arguments.size());
  for(const auto& argument:arguments)descriptors.push_back(argument.desc);
  const int64_t integers[]={mode,dim,k,int64_t(descending),int64_t(ordered),int64_t(stable),int64_t(self.dim()==0)};
  check(sorting_native(descriptors.data(),descriptors.size(),integers,7));
  return {values,indices};
}

std::tuple<at::Tensor,at::Tensor> topk_tensor(const at::Tensor& self,c10::SymInt k,int64_t dim,bool largest,bool sorted) {
  return sorting_result(self,0,dim,k.expect_int(),largest,sorted,false);
}
std::tuple<at::Tensor,at::Tensor> sort_tensor(const at::Tensor& self,int64_t dim,bool descending) {
  return sorting_result(self,1,dim,0,descending,true,false);
}
std::tuple<at::Tensor,at::Tensor> sort_stable_tensor(const at::Tensor& self,std::optional<bool> stable,int64_t dim,bool descending) {
  return sorting_result(self,1,dim,0,descending,true,stable.value_or(false));
}
at::Tensor argsort_tensor(const at::Tensor& self,int64_t dim,bool descending) {
  return std::get<1>(sorting_result(self,2,dim,0,descending,true,false));
}
at::Tensor argsort_stable_tensor(const at::Tensor& self,bool stable,int64_t dim,bool descending) {
  return std::get<1>(sorting_result(self,2,dim,0,descending,true,stable));
}

void sorting_output_contract(const at::Tensor& self,const at::Tensor& out,c10::ScalarType dtype) {
  TORCH_CHECK(out.device()==self.device() && out.scalar_type()==dtype && out.layout()==c10::Layout::Strided,
              "sort output device, dtype or layout differs");
  TORCH_CHECK(!out.is_conj() && !out.is_neg(),"sort output must have resolved view bits");
}
std::tuple<at::Tensor&,at::Tensor&> sorting_copy_result(std::tuple<at::Tensor,at::Tensor> result,at::Tensor& values,at::Tensor& indices) {
  at::native::resize_output(values,std::get<0>(result).sizes());
  at::native::resize_output(indices,std::get<1>(result).sizes());
  at::assert_no_internal_overlap(values);at::assert_no_internal_overlap(indices);at::assert_no_overlap(values,indices);
  values.copy_(std::get<0>(result));indices.copy_(std::get<1>(result));
  return std::forward_as_tuple(values,indices);
}
std::tuple<at::Tensor&,at::Tensor&> topk_tensor_out(const at::Tensor& self,c10::SymInt k,int64_t dim,bool largest,bool sorted,at::Tensor& values,at::Tensor& indices) {
  sorting_output_contract(self,values,self.scalar_type());sorting_output_contract(self,indices,at::kLong);
  return sorting_copy_result(topk_tensor(self,k,dim,largest,sorted),values,indices);
}
std::tuple<at::Tensor&,at::Tensor&> sort_tensor_out(const at::Tensor& self,int64_t dim,bool descending,at::Tensor& values,at::Tensor& indices) {
  sorting_output_contract(self,values,self.scalar_type());sorting_output_contract(self,indices,at::kLong);
  return sorting_copy_result(sort_tensor(self,dim,descending),values,indices);
}
std::tuple<at::Tensor&,at::Tensor&> sort_stable_tensor_out(const at::Tensor& self,std::optional<bool> stable,int64_t dim,bool descending,at::Tensor& values,at::Tensor& indices) {
  sorting_output_contract(self,values,self.scalar_type());sorting_output_contract(self,indices,at::kLong);
  return sorting_copy_result(sort_stable_tensor(self,stable,dim,descending),values,indices);
}
at::Tensor& argsort_stable_tensor_out(const at::Tensor& self,bool stable,int64_t dim,bool descending,at::Tensor& out) {
  sorting_output_contract(self,out,at::kLong);
  auto result=argsort_stable_tensor(self,stable,dim,descending);
  at::native::resize_output(out,result.sizes());at::assert_no_internal_overlap(out);out.copy_(result);
  return out;
}
