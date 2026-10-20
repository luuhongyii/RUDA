using KdaForward = int (*)(const Descriptor*, size_t, float, uint32_t, bool);
using KdaBackward = int (*)(const Descriptor*, size_t, float, uint32_t);
KdaForward kda_forward_native = nullptr;
KdaBackward kda_backward_native = nullptr;

void kda_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& beta, const at::Tensor& decay, const at::Tensor& initial,
    double scale, int64_t chunk) {
  for (const auto& t : {q,k,v,beta,decay,initial}) {
    training_tensor(t);
    TORCH_CHECK(t.device()==q.device(), "KDA operands must share a device");
    TORCH_CHECK(uint64_t(t.numel())<=std::numeric_limits<uint32_t>::max(), "KDA exceeds uint32 indexing");
  }
  TORCH_CHECK(q.dim()==4 && q.sizes()==k.sizes() && v.dim()==4 && beta.dim()==3 &&
              decay.sizes()==q.sizes() && initial.dim()==4 && std::isfinite(scale) &&
              std::isfinite(float(scale)) && chunk>0 && uint64_t(chunk)<=std::numeric_limits<uint32_t>::max(),
              "invalid channel-gated KDA inputs");
  TORCH_CHECK(k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type() &&
              beta.scalar_type()==at::kFloat && decay.scalar_type()==at::kFloat && initial.scalar_type()==at::kFloat,
              "KDA requires matching Q/K/V storage and FP32 beta/decay/state");
  TORCH_CHECK(q.size(1)>0 && q.size(3)>0 && v.size(3)>0, "KDA head dimensions must be positive");
  for (int64_t i=0;i<3;++i)
    TORCH_CHECK(q.size(i)==v.size(i) && q.size(i)==beta.size(i), "KDA batch/head/sequence mismatch");
  TORCH_CHECK(initial.size(0)==q.size(0) && initial.size(1)==q.size(1) &&
              initial.size(2)==q.size(3) && initial.size(3)==v.size(3), "KDA state shape mismatch");
}

std::vector<int64_t> kda_boundary_shape(const at::Tensor& q, const at::Tensor& v, int64_t chunk, bool retain) {
  return {retain ? q.size(2)/chunk + (q.size(2)%chunk!=0) : 0, q.size(0),q.size(1),q.size(3),v.size(3)};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> kda_forward(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& beta, const at::Tensor& decay, const at::Tensor& initial,
    double scale, int64_t chunk, bool retain) {
  TORCH_CHECK(kda_forward_native, "RUDA KDA API 1 unavailable; rebuild both native libraries");
  kda_inputs(q,k,v,beta,decay,initial,scale,chunk);
  auto shape=kda_boundary_shape(q,v,chunk,retain);
  uint64_t elements=1;
  for (auto size:shape) {
    TORCH_CHECK(size>=0 && uint64_t(size)<=std::numeric_limits<uint32_t>::max(), "KDA boundary dimension overflow");
    elements*=uint64_t(size);
    TORCH_CHECK(elements<=std::numeric_limits<uint32_t>::max(), "KDA boundaries exceed uint32 indexing");
  }
  auto output=at::empty(v.sizes(),v.options());
  auto final=at::empty(initial.sizes(),initial.options());
  auto boundaries=at::empty(shape,initial.options());
  std::vector<Argument> args; args.reserve(9);
  for (const auto& t:{q,k,v,beta,decay,initial,output,final,boundaries}) args.emplace_back(t);
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for (const auto& arg:args) ds.push_back(arg.desc);
  check(kda_forward_native(ds.data(),ds.size(),float(scale),uint32_t(chunk),retain));
  return {output,final,boundaries};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> kda_backward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& beta,
    const at::Tensor& decay, const at::Tensor& initial, const at::Tensor& boundaries,
    const at::Tensor& dy, const at::Tensor& dh, double scale, int64_t chunk) {
  TORCH_CHECK(kda_backward_native, "RUDA KDA API 1 unavailable; rebuild both native libraries");
  kda_inputs(q,k,v,beta,decay,initial,scale,chunk);
  training_same(v,dy); training_same(initial,dh); training_tensor(boundaries);
  TORCH_CHECK(boundaries.device()==q.device() && boundaries.scalar_type()==at::kFloat &&
              boundaries.sizes().vec()==kda_boundary_shape(q,v,chunk,true), "invalid KDA boundaries");
  auto dq=at::empty(q.sizes(),q.options()); auto dk=at::empty(k.sizes(),k.options());
  auto dv=at::empty(v.sizes(),v.options()); auto db=at::empty(beta.sizes(),beta.options());
  auto dg=at::empty(decay.sizes(),decay.options()); auto ds0=at::empty(initial.sizes(),initial.options());
  std::vector<Argument> args; args.reserve(15);
  for (const auto& t:{q,k,v,beta,decay,initial,boundaries,dy,dh,dq,dk,dv,db,dg,ds0}) args.emplace_back(t);
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for (const auto& arg:args) ds.push_back(arg.desc);
  check(kda_backward_native(ds.data(),ds.size(),float(scale),uint32_t(chunk)));
  return {dq,dk,dv,db,dg,ds0};
}
