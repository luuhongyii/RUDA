void scientific_call(uint32_t op,const std::vector<at::Tensor>& tensors,
                     const std::vector<int64_t>& integers,const std::vector<double>& scalars) {
  TORCH_CHECK(scientific_native,"RUDA scientific API 1 requires matching rebuilt native libraries");
  std::vector<Argument> arguments;arguments.reserve(tensors.size());
  std::vector<Descriptor> descriptors;descriptors.reserve(tensors.size());
  for(const auto& tensor:tensors)arguments.emplace_back(tensor,op==207);
  for(const auto& argument:arguments)descriptors.push_back(argument.desc);
  check(scientific_native(op,descriptors.data(),descriptors.size(),integers.data(),integers.size(),scalars.data(),scalars.size()));
}
void scientific_fill(const at::Tensor& output,double re,double im) {
  scientific_call(205,{output},{},{re,im});
}
at::Tensor scientific_resolve(const at::Tensor& value) {
  if(!value.is_conj() && !value.is_neg())return value;
  auto output=at::empty(value.sizes(),value.options());
  scientific_call(207,{value,output},{int64_t(value.is_conj()),int64_t(value.is_neg())},{});
  return output;
}
