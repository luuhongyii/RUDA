using AdvancedIndex = int (*)(const Descriptor*,const Descriptor*,size_t,const int64_t*,const int64_t*,const Descriptor*);
AdvancedIndex advanced_index_native = nullptr;

struct IndexMetadata {
  at::Tensor source;
  std::vector<at::Tensor> indices;
  std::vector<int64_t> sizes, strides;
  int64_t before = 0;
  std::vector<int64_t> replacement;
};

template <typename Expand>
std::vector<at::Tensor> expand_indices_on_device(Expand expand, const at::Tensor& source,
    const c10::List<std::optional<at::Tensor>>& original) {
  if constexpr (std::is_invocable_v<Expand, decltype(source), decltype(original), bool>) {
    return expand(source, original, true);
  } else {
    auto indices = expand(source, original);
    for (auto& index : indices) {
      if (index.defined() && index.device() != source.device()) index = index.to(source.device());
    }
    return indices;
  }
}

IndexMetadata index_metadata(at::Tensor source,const c10::List<std::optional<at::Tensor>>& original) {
  TORCH_CHECK_INDEX(original.size()<=size_t(source.dim()),"too many indices for tensor of dimension ",source.dim());
  for (const auto& item:original) {
    const std::optional<at::Tensor> index=item;
    TORCH_CHECK(!index || index->device().is_cpu() || index->device()==source.device(),
                "indices should be either on cpu or on the same device as the indexed tensor");
  }
  at::native::checkIndexTensorTypes(original,true);
  auto indices=expand_indices_on_device(&at::native::expandTensors,source,original);
  try {
    indices=at::expand_outplace(indices);
  } catch (const std::exception&) {
    TORCH_CHECK_INDEX(false,"shape mismatch: indexing tensors could not be broadcast together");
  }
  while (indices.size()<size_t(source.dim())) indices.emplace_back();
  if (!at::native::hasContiguousSubspace(indices))
    std::tie(source,indices)=at::native::transposeToFront(source,indices);
  IndexMetadata info;
  int64_t before=0,after=0,indexed=0;
  std::vector<int64_t> replacement;
  for (size_t dim=0;dim<indices.size();++dim) {
    if (!indices[dim].defined()) {
      if (!indexed) ++before; else ++after;
    } else {
      if (indices[dim].scalar_type()==at::kInt) indices[dim]=indices[dim].to(at::kLong);
      ++indexed;
      replacement=indices[dim].sizes().vec();
      info.sizes.push_back(source.size(dim));
      info.strides.push_back(source.stride(dim)*source.element_size());
    }
  }
  TORCH_CHECK_INDEX(std::find(info.sizes.begin(),info.sizes.end(),0)==info.sizes.end()
      || std::find(replacement.begin(),replacement.end(),0)!=replacement.end(),
      "index is out of bounds for dimension with size 0");
  auto shape=source.sizes().vec(),strides=source.strides().vec();
  shape.erase(shape.begin()+before,shape.begin()+before+indexed);
  strides.erase(strides.begin()+before,strides.begin()+before+indexed);
  shape.insert(shape.begin()+before,replacement.begin(),replacement.end());
  strides.insert(strides.begin()+before,replacement.size(),0);
  info.source=source.as_strided(shape,strides);
  info.before=before;
  info.replacement=replacement;
  for (auto& index:indices) {
    if (!index.defined()) continue;
    std::vector<int64_t> index_shape(before,1);
    index_shape.insert(index_shape.end(),index.sizes().begin(),index.sizes().end());
    index_shape.insert(index_shape.end(),after,1);
    info.indices.push_back(index.reshape(index_shape).expand(shape));
  }
  return info;
}

void index_dispatch(const IndexMetadata& info,at::Tensor& result) {
  TORCH_CHECK(advanced_index_native,"RUDA indexing API 1 unavailable; rebuild both native libraries");
  Argument source(info.source),out(result);
  std::vector<Argument> args;
  args.reserve(info.indices.size());
  for (const auto& index:info.indices) args.emplace_back(index);
  std::vector<Descriptor> descriptors;
  for (const auto& arg:args) descriptors.push_back(arg.desc);
  check(advanced_index_native(&source.desc,descriptors.data(),descriptors.size(),
                             info.sizes.data(),info.strides.data(),&out.desc));
}

at::Tensor advanced_index(const at::Tensor& self,const c10::List<std::optional<at::Tensor>>& indices) {
  validate(self.device());
  auto info=index_metadata(self,indices);
  auto result=at::empty(info.source.sizes(),self.options());
  index_dispatch(info,result);
  return result;
}

at::Tensor& advanced_index_out(const at::Tensor& self,const c10::List<std::optional<at::Tensor>>& indices,
                              at::Tensor& out) {
  validate(self.device());
  TORCH_CHECK(out.device()==self.device() && out.scalar_type()==self.scalar_type(),
              "index_out: self and result must have the same dtype and device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out,self);
  for (const auto& item:indices) {
    const std::optional<at::Tensor> index=item;
    if (index && index->defined()) at::assert_no_overlap(out,*index);
  }
  auto info=index_metadata(self,indices);
  at::native::resize_output(out,info.source.sizes());
  index_dispatch(info,out);
  return out;
}
