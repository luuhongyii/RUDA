using BlasCommand = int (*)(uint32_t,const Descriptor*,size_t,const int64_t*,size_t,const double*,size_t);
BlasCommand blas_native=nullptr;

void blas_call(uint32_t op,const std::vector<at::Tensor>& tensors,
               const std::vector<int64_t>& ints={},const std::vector<double>& scalars={}) {
  TORCH_CHECK(blas_native,"RUDA BLAS API 1 unavailable; rebuild Rust and C++ extensions");
  std::vector<Argument> args;args.reserve(tensors.size());
  for(const auto& t:tensors)args.emplace_back(t);
  std::vector<Descriptor> ds;ds.reserve(args.size());for(const auto& a:args)ds.push_back(a.desc);
  check(blas_native(op,ds.data(),ds.size(),ints.data(),ints.size(),scalars.data(),scalars.size()));
}
void blas_same_device(const at::Tensor& a,const at::Tensor& b) {
  validate(a.device());validate(b.device());TORCH_CHECK(a.device()==b.device(),"BLAS operands must share a RUDA device");
}
std::vector<int64_t> blas_batch(const at::Tensor& a,const at::Tensor& b) {
  auto rank=std::max(a.dim(),b.dim());std::vector<int64_t> shape(rank,1);
  for(int64_t i=0;i<rank-2;++i) {
    auto ad=i+a.dim()>=rank?a.size(i+a.dim()-rank):1;
    auto bd=i+b.dim()>=rank?b.size(i+b.dim()-rank):1;
    TORCH_CHECK(ad==bd||ad==1||bd==1,"BLAS leading batches do not broadcast");shape[i]=ad==1?bd:ad;
  }return shape;
}
at::Tensor blas_gemm(const at::Tensor& a,const at::Tensor& b,const std::optional<at::Tensor>& c,
    double ar,double ai,double br,double bi,bool ta,bool tb,bool ca,bool cb,int64_t math,int64_t strategy) {
  blas_same_device(a,b);TORCH_CHECK(a.dim()>=2&&b.dim()>=2,"GEMM requires matrix operands");
  TORCH_CHECK(a.scalar_type()==b.scalar_type()&&(a.scalar_type()==at::kFloat||a.scalar_type()==at::kDouble||
    a.scalar_type()==at::kComplexFloat||a.scalar_type()==at::kComplexDouble),"GEMM requires matching F32/F64/complex operands");
  if(a.scalar_type()==at::kFloat||a.scalar_type()==at::kComplexFloat) {
    ar=double(float(ar));ai=double(float(ai));br=double(float(br));bi=double(float(bi));
  }
  auto m=a.size(a.dim()-(ta?1:2)),k=a.size(a.dim()-(ta?2:1));
  auto bk=b.size(b.dim()-(tb?1:2)),n=b.size(b.dim()-(tb?2:1));
  TORCH_CHECK(k==bk,"GEMM contraction sizes differ");auto shape=blas_batch(a,b);shape[shape.size()-2]=m;shape.back()=n;
  auto out=at::empty(shape,a.options());std::vector<at::Tensor> tensors={a,b};
  if(c) {blas_same_device(a,*c);TORCH_CHECK(c->scalar_type()==a.scalar_type()&&c->sizes()==out.sizes(),"GEMM C must match output shape and dtype");tensors.push_back(*c);}
  TORCH_CHECK(c|| (br==0.&&bi==0.),"nonzero beta requires C");tensors.push_back(out);
  blas_call(0,tensors,{ta,tb,ca,cb,math,strategy},{ar,ai,br,bi});return out;
}
at::Tensor blas_fused(const at::Tensor& a,const at::Tensor& b,const at::Tensor& bias,
    int64_t activation,int64_t strategy,int64_t output_code) {
  blas_same_device(a,b);blas_same_device(a,bias);
  TORCH_CHECK(a.dim()==2&&b.dim()==2&&bias.dim()==1&&a.size(1)==b.size(0)&&bias.size(0)==b.size(1),"fused GEMM expects [M,K],[K,N],bias[N]");
  TORCH_CHECK((a.scalar_type()==at::kHalf||a.scalar_type()==at::kBFloat16)&&b.scalar_type()==a.scalar_type(),"fused GEMM requires matching FP16/BF16 inputs");
  TORCH_CHECK(output_code==0||output_code==1,"fused GEMM output code must be input(0) or FP32(1)");
  auto output_dtype=output_code==1?at::kFloat:a.scalar_type();
  TORCH_CHECK((bias.scalar_type()==a.scalar_type()||bias.scalar_type()==at::kFloat)&&
    (output_dtype==a.scalar_type()||output_dtype==at::kFloat),"fused GEMM bias/output requires input storage or FP32");
  auto out=at::empty({a.size(0),b.size(1)},a.options().dtype(output_dtype));
  blas_call(1,{a,b,bias,out},{activation,strategy});return out;
}
at::Tensor blas_grouped(const at::Tensor& x,const at::Tensor& w,const at::Tensor& ids,int64_t strategy) {
  blas_same_device(x,w);blas_same_device(x,ids);
  TORCH_CHECK(x.dim()==2&&w.dim()==3&&ids.dim()==1&&ids.size(0)==x.size(0)&&w.size(2)==x.size(1),"grouped GEMM expects [M,K],[E,N,K],ids[M]");
  TORCH_CHECK(w.scalar_type()==x.scalar_type()&&(x.scalar_type()==at::kFloat||x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16),"grouped GEMM requires matching FP32/FP16/BF16 inputs");
  TORCH_CHECK(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong,"expert IDs require int32/int64");
  auto out=at::empty({x.size(0),w.size(1)},x.options());blas_call(2,{x,w,ids,out},{strategy});return out;
}
std::tuple<at::Tensor,at::Tensor> blas_grouped_backward(const at::Tensor& x,const at::Tensor& w,
    const at::Tensor& ids,const at::Tensor& grad) {
  blas_same_device(x,w);blas_same_device(x,ids);blas_same_device(x,grad);
  TORCH_CHECK(x.dim()==2&&w.dim()==3&&ids.dim()==1&&ids.size(0)==x.size(0)&&w.size(2)==x.size(1)&&
    grad.dim()==2&&grad.size(0)==x.size(0)&&grad.size(1)==w.size(1),"invalid grouped backward geometry");
  TORCH_CHECK(x.scalar_type()==w.scalar_type()&&x.scalar_type()==grad.scalar_type(),"grouped backward precision mismatch");
  TORCH_CHECK(x.scalar_type()==at::kFloat||x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16,"grouped backward requires FP32/FP16/BF16");
  TORCH_CHECK(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong,"expert IDs require int32/int64");
  auto dx=at::empty(x.sizes(),x.options()),dw=at::empty(w.sizes(),w.options());blas_call(3,{x,w,ids,grad,dx,dw});return {dx,dw};
}
void indexed_expert_metadata(const at::Tensor& x,const std::vector<at::Tensor>& weights,const at::Tensor& ids,int64_t start) {
  blas_same_device(x,ids);
  TORCH_CHECK(x.dim()==2 && x.size(1)>0 && ids.dim()==1 && ids.size(0)==x.size(0),"invalid indexed expert input/IDs");
  TORCH_CHECK(!weights.empty() && weights.size()<=16 && start>=0 && start<=int64_t(UINT32_MAX)-int64_t(weights.size()),"invalid indexed expert bank");
  TORCH_CHECK((x.scalar_type()==at::kFloat||x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16)
    && (ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong),"invalid indexed expert storage");
  for(const auto& w:weights) {
    blas_same_device(x,w);
    TORCH_CHECK(w.dim()==2 && w.size(0)>0 && w.size(1)==x.size(1) && w.sizes()==weights[0].sizes()
      && w.scalar_type()==x.scalar_type(),"indexed expert weights differ");
  }
}
at::Tensor indexed_expert_linear(const at::Tensor& x,const std::vector<at::Tensor>& weights,const at::Tensor& ids,int64_t start) {
  indexed_expert_metadata(x,weights,ids,start);
  auto out=at::empty({x.size(0),weights[0].size(0)},x.options());
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()};
  for(const auto& w:weights)tensors.push_back(w.contiguous());
  tensors.push_back(out);blas_call(7,tensors,{start,int64_t(weights.size())});return out;
}
std::vector<at::Tensor> indexed_expert_backward(const at::Tensor& x,const std::vector<at::Tensor>& weights,
    const at::Tensor& ids,const at::Tensor& grad,int64_t start) {
  indexed_expert_metadata(x,weights,ids,start);blas_same_device(x,grad);
  TORCH_CHECK(grad.dim()==2 && grad.size(0)==x.size(0) && grad.size(1)==weights[0].size(0)
    && grad.scalar_type()==x.scalar_type(),"indexed expert gradient differs");
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()},outputs={at::empty(x.sizes(),x.options())};
  for(const auto& w:weights) {tensors.push_back(w.contiguous());outputs.push_back(at::empty(w.sizes(),w.options()));}
  tensors.push_back(grad.contiguous());tensors.insert(tensors.end(),outputs.begin(),outputs.end());
  blas_call(8,tensors,{start,int64_t(weights.size())});return outputs;
}
at::Tensor blas_triangular(const at::Tensor& a,const at::Tensor& b,bool left,bool upper,bool transpose,bool unit,double alpha) {
  blas_same_device(a,b);TORCH_CHECK(a.dim()>=2&&b.dim()>=2&&a.size(-1)==a.size(-2),"triangular solve requires square matrix A and matrix B");
  TORCH_CHECK(a.scalar_type()==b.scalar_type()&&(a.scalar_type()==at::kFloat||a.scalar_type()==at::kDouble),"triangular solve requires matching F32/F64");
  TORCH_CHECK(a.size(-1)==b.size(left?-2:-1),"triangular solve order mismatch");auto shape=blas_batch(a,b);
  shape[shape.size()-2]=b.size(-2);shape.back()=b.size(-1);auto out=at::empty(shape,b.options());
  blas_call(4,{a,b,out},{left,upper,transpose,unit},{alpha});return out;
}

std::vector<at::Tensor> indexed_expert_backward_selected(const at::Tensor& x,const std::vector<at::Tensor>& weights,
    const at::Tensor& ids,const at::Tensor& grad,int64_t start,bool need_x,const std::vector<bool>& needed) {
  indexed_expert_metadata(x,weights,ids,start);blas_same_device(x,grad);
  TORCH_CHECK(needed.size()==weights.size() && grad.dim()==2 && grad.size(0)==x.size(0)
    && grad.size(1)==weights[0].size(0) && grad.scalar_type()==x.scalar_type(),"indexed expert selected gradient differs");
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()};
  std::vector<at::Tensor> outputs={at::empty(need_x?x.sizes().vec():std::vector<int64_t>{0},x.options())};
  int64_t mask=0;
  for(size_t i=0;i<weights.size();++i) {
    tensors.push_back(weights[i].contiguous());
    outputs.push_back(at::empty(needed[i]?weights[i].sizes().vec():std::vector<int64_t>{0},weights[i].options()));
    if(needed[i])mask|=int64_t(1)<<i;
  }
  tensors.push_back(grad.contiguous());tensors.insert(tensors.end(),outputs.begin(),outputs.end());
  blas_call(9,tensors,{start,int64_t(weights.size()),int64_t(need_x),mask});return outputs;
}

std::vector<at::Tensor> indexed_block_float8_statistics(const at::Tensor& x,const std::vector<at::Tensor>& weights,
    const at::Tensor& ids,const at::Tensor& gradient,int64_t start,int64_t mode,const std::vector<int64_t>& geometry,
    const std::vector<int64_t>& capacities) {
  blas_same_device(x,ids);blas_same_device(x,gradient);
  TORCH_CHECK(!weights.empty() && weights.size()<=16 && geometry.size()==6 && capacities.size()==2
    && capacities[0]>=0 && capacities[1]>=0,"invalid block FP8 statistics signature");
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()},outputs;
  for(const auto& w:weights) {blas_same_device(x,w);tensors.push_back(w.contiguous());}
  tensors.push_back(gradient.contiguous());
  for(auto capacity:capacities) {outputs.push_back(at::empty({2,capacity},x.options().dtype(at::kFloat)));tensors.push_back(outputs.back());}
  std::vector<int64_t> policy={start,int64_t(weights.size()),mode};policy.insert(policy.end(),geometry.begin(),geometry.end());
  blas_call(14,tensors,policy);return outputs;
}

std::vector<at::Tensor> indexed_block_float8_product(const at::Tensor& x,const std::vector<at::Tensor>& weights,
    const at::Tensor& ids,const at::Tensor& gradient,const std::vector<at::Tensor>& statistics,int64_t start,int64_t mode,
    const std::vector<int64_t>& geometry,const std::vector<int64_t>& formats,int64_t margin,bool power,int64_t strategy,
    const std::vector<bool>& needed) {
  blas_same_device(x,ids);blas_same_device(x,gradient);
  TORCH_CHECK(x.dim()==2 && !weights.empty() && weights.size()<=16 && geometry.size()==6 && statistics.size()==2
    && formats.size()==2 && needed.size()==weights.size() && mode>=0 && mode<=2,"invalid block FP8 product signature");
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()},outputs;int64_t mask=0;
  for(size_t i=0;i<weights.size();++i) {
    const auto& w=weights[i];blas_same_device(x,w);TORCH_CHECK(w.dim()==2,"FP8 expert weight must be a matrix");
    tensors.push_back(w.contiguous());if(needed[i])mask|=int64_t(1)<<i;
    if(mode==2)outputs.push_back(at::empty(needed[i]?w.sizes().vec():std::vector<int64_t>{0},w.options().dtype(at::kFloat)));
  }
  tensors.push_back(gradient.contiguous());for(const auto& stat:statistics) {blas_same_device(x,stat);tensors.push_back(stat.contiguous());}
  if(mode!=2)outputs.push_back(at::empty({x.size(0),weights[0].size(mode==0?0:1)},x.options().dtype(at::kFloat)));
  tensors.insert(tensors.end(),outputs.begin(),outputs.end());
  std::vector<int64_t> policy={start,int64_t(weights.size()),mode};policy.insert(policy.end(),geometry.begin(),geometry.end());
  policy.insert(policy.end(),{formats[0],formats[1],margin,int64_t(power),strategy,mask});blas_call(15,tensors,policy);return outputs;
}

at::Tensor indexed_float8_statistics(const at::Tensor& x,const std::vector<at::Tensor>& weights,
    const at::Tensor& ids,int64_t start,bool include_weights) {
  blas_same_device(x,ids);TORCH_CHECK(!weights.empty() && weights.size()<=16,"invalid FP8 expert bank");
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()};
  for(const auto& w:weights) {blas_same_device(x,w);tensors.push_back(w.contiguous());}
  auto output=at::empty({int64_t(weights.size()),2,3},x.options().dtype(at::kFloat));tensors.push_back(output);
  blas_call(12,tensors,{start,int64_t(weights.size()),int64_t(include_weights)});return output;
}

std::vector<at::Tensor> indexed_float8_product(const at::Tensor& x,const std::vector<at::Tensor>& weights,
    const at::Tensor& ids,const at::Tensor& gradient,const at::Tensor& scales,int64_t start,int64_t mode,
    const std::vector<int64_t>& formats,int64_t strategy,const std::vector<bool>& needed) {
  blas_same_device(x,ids);blas_same_device(x,gradient);blas_same_device(x,scales);
  TORCH_CHECK(x.dim()==2 && !weights.empty() && weights.size()<=16 && formats.size()==2
    && needed.size()==weights.size() && mode>=0 && mode<=2,"invalid FP8 expert product signature");
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()},outputs;int64_t mask=0;
  for(size_t i=0;i<weights.size();++i) {
    const auto& w=weights[i];blas_same_device(x,w);TORCH_CHECK(w.dim()==2,"FP8 expert weight must be a matrix");
    tensors.push_back(w.contiguous());if(needed[i])mask|=int64_t(1)<<i;
    if(mode==2)outputs.push_back(at::empty(needed[i]?w.sizes().vec():std::vector<int64_t>{0},w.options().dtype(at::kFloat)));
  }
  if(mode!=2)outputs.push_back(at::empty({x.size(0),weights[0].size(mode==0?0:1)},x.options().dtype(at::kFloat)));
  tensors.push_back(gradient.contiguous());tensors.push_back(scales.contiguous());tensors.insert(tensors.end(),outputs.begin(),outputs.end());
  blas_call(13,tensors,{start,int64_t(weights.size()),mode,formats[0],formats[1],strategy,mask});return outputs;
}

at::Tensor indexed_quantized_expert_linear(const at::Tensor& x,const std::vector<at::Tensor>& buffers,
    const at::Tensor& ids,int64_t start,const std::vector<int64_t>& layouts,bool transpose) {
  blas_same_device(x,ids);
  TORCH_CHECK(x.dim()==2 && ids.dim()==1 && ids.size(0)==x.size(0),"invalid quantized expert routed rows");
  TORCH_CHECK(!layouts.empty() && layouts.size()%6==0 && layouts.size()/6<=16
    && buffers.size()==layouts.size()/6*4,"invalid quantized expert buffer/layout count");
  auto width=layouts[transpose?2:3];TORCH_CHECK(width>0,"invalid quantized expert output width");
  auto output=at::empty({x.size(0),width},x.options());
  std::vector<at::Tensor> tensors={x.contiguous(),ids.contiguous()};
  for(const auto& buffer:buffers) {blas_same_device(x,buffer);tensors.push_back(buffer.contiguous());}
  tensors.push_back(output);
  std::vector<int64_t> policy={start,int64_t(layouts.size()/6),int64_t(transpose)};
  policy.insert(policy.end(),layouts.begin(),layouts.end());blas_call(11,tensors,policy);return output;
}

at::Tensor indexed_expert_activity(const at::Tensor& ids,int64_t start,int64_t count) {
  validate(ids.device());TORCH_CHECK(ids.dim()==1 && count>0,"invalid indexed expert activity geometry");
  auto result=at::empty({count},ids.options().dtype(at::kInt));
  blas_call(10,{ids.contiguous(),result},{start,count});return result;
}
at::Tensor blas_reduce_to(const at::Tensor& input,const std::vector<int64_t>& shape,bool conjugate) {
  validate(input.device());TORCH_CHECK(input.dim()>=int64_t(shape.size()),"cannot reduce to a larger rank");
  for(size_t i=0;i<shape.size();++i)TORCH_CHECK(shape[i]>=0&&(shape[i]==1||shape[i]==input.size(input.dim()-shape.size()+i)),"invalid sum-to shape");
  auto out=at::empty(shape,input.options());blas_call(5,{input,out},{conjugate});return out;
}
at::Tensor blas_scale(const at::Tensor& input,double real,double imag) {
  validate(input.device());auto out=at::empty(input.sizes(),input.options());blas_call(6,{input,out},{},{real,imag});return out;
}
