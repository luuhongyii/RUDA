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
at::Tensor blas_triangular(const at::Tensor& a,const at::Tensor& b,bool left,bool upper,bool transpose,bool unit,double alpha) {
  blas_same_device(a,b);TORCH_CHECK(a.dim()>=2&&b.dim()>=2&&a.size(-1)==a.size(-2),"triangular solve requires square matrix A and matrix B");
  TORCH_CHECK(a.scalar_type()==b.scalar_type()&&(a.scalar_type()==at::kFloat||a.scalar_type()==at::kDouble),"triangular solve requires matching F32/F64");
  TORCH_CHECK(a.size(-1)==b.size(left?-2:-1),"triangular solve order mismatch");auto shape=blas_batch(a,b);
  shape[shape.size()-2]=b.size(-2);shape.back()=b.size(-1);auto out=at::empty(shape,b.options());
  blas_call(4,{a,b,out},{left,upper,transpose,unit},{alpha});return out;
}
at::Tensor blas_reduce_to(const at::Tensor& input,const std::vector<int64_t>& shape,bool conjugate) {
  validate(input.device());TORCH_CHECK(input.dim()>=int64_t(shape.size()),"cannot reduce to a larger rank");
  for(size_t i=0;i<shape.size();++i)TORCH_CHECK(shape[i]>=0&&(shape[i]==1||shape[i]==input.size(input.dim()-shape.size()+i)),"invalid sum-to shape");
  auto out=at::empty(shape,input.options());blas_call(5,{input,out},{conjugate});return out;
}
at::Tensor blas_scale(const at::Tensor& input,double real,double imag) {
  validate(input.device());auto out=at::empty(input.sizes(),input.options());blas_call(6,{input,out},{},{real,imag});return out;
}
