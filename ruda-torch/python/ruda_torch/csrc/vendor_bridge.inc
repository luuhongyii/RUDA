using VendorCallback = int (*)(uintptr_t,const uintptr_t*,size_t,uintptr_t);
using VendorScope = int (*)(VendorCallback,uintptr_t,uint64_t,const Descriptor*,size_t);
VendorScope vendor_scope_native=nullptr;

struct VendorRequest {
  uint32_t op;
  std::vector<int64_t> ints;
  std::vector<double> scalars;
  std::unordered_map<std::string,uintptr_t> functions;
  std::vector<uint64_t> results;
  std::string error;
  template<class... Args> void call(const std::string& name,Args... args) const {
    auto found=functions.find(name);
    if(found==functions.end() || !found->second)throw std::runtime_error("missing CUDA symbol: "+name);
    auto fn=reinterpret_cast<int (*)(Args...)>(found->second);
    int status=fn(args...);
    if(status)throw std::runtime_error(name+" failed with status="+std::to_string(status));
  }
};

struct VendorSession {
  VendorRequest& request;
  void* stream;
  void* handle=nullptr;
  std::string destroy;
  std::vector<uint64_t> allocations;
  bool finished=false;
  VendorSession(VendorRequest& r,uintptr_t s):request(r),stream(reinterpret_cast<void*>(s)) {}
  void create(bool blas) {
    destroy=blas?"cublasDestroy_v2":"cusolverDnDestroy";
    request.call(blas?"cublasCreate_v2":"cusolverDnCreate",&handle);
    request.call(blas?"cublasSetStream_v2":"cusolverDnSetStream",handle,stream);
    if(blas)request.call("cublasSetPointerMode_v2",handle,0);
  }
  void* alloc(size_t bytes) {
    uint64_t ptr=0;request.call("cuMemAlloc_v2",&ptr,std::max(size_t(1),bytes));
    allocations.push_back(ptr);return reinterpret_cast<void*>(ptr);
  }
  void* work(int length,int width) {
    if(length<0 || width<=0 || size_t(length)>std::numeric_limits<size_t>::max()/size_t(width))
      throw std::runtime_error("invalid vendor workspace size");
    return alloc(size_t(length)*size_t(width));
  }
  void finish() {
    if(finished)return;
    request.call("cuStreamSynchronize",stream);
    if(handle) {request.call(destroy,handle);handle=nullptr;}
    while(!allocations.empty()) {request.call("cuMemFree_v2",allocations.back());allocations.pop_back();}
    finished=true;
  }
  ~VendorSession() {
    // Never unwind through the Rust callback ABI. The outer scope also drains.
    try {finish();} catch(const std::exception& e) {
      request.error+=(request.error.empty()?"":"; ")+std::string("cleanup: ")+e.what();
    }
  }
};

int vendor_callback(uintptr_t raw_stream,const uintptr_t* pointers,size_t count,uintptr_t opaque) noexcept {
  auto& r=*reinterpret_cast<VendorRequest*>(opaque);
  try {
    auto v=[&](size_t i)->int {auto x=r.ints.at(i);if(x<0 || x>INT32_MAX)throw std::runtime_error("vendor int32 parameter overflow");return int(x);};
    auto ptr=[&](size_t i,uint64_t offset=0)->void* {if(i>=count)throw std::runtime_error("missing vendor tensor");return reinterpret_cast<void*>(pointers[i]+offset);};
    VendorSession session(r,raw_stream);
    size_t wi=r.op==4 || r.op==9?2:r.op==7?4:r.op==8?6:r.op==10 || r.op==11?5:3;
    const auto prefix=std::string("cusolverDn")+((r.op>=2 && r.op<=11 && v(wi)==8)?"D":"S");
    if(r.op==0) {
      int m=v(0),n=v(1),k=v(2),batch=v(3),type=v(4),out_type=v(5),compute=v(6);
      session.create(true);
      float af[]={float(r.scalars.at(0)),float(r.scalars.at(1))},bf[]={float(r.scalars.at(2)),float(r.scalars.at(3))};
      double ad[]={r.scalars.at(0),r.scalars.at(1)},bd[]={r.scalars.at(2),r.scalars.at(3)};
      const void* a=type==1 || type==5?static_cast<const void*>(ad):static_cast<const void*>(af);
      const void* b=type==1 || type==5?static_cast<const void*>(bd):static_cast<const void*>(bf);
      r.call("cublasGemmStridedBatchedEx",session.handle,0,0,m,n,k,a,ptr(0),type,std::max(1,m),static_cast<long long>(m)*k,
        ptr(1),type,std::max(1,k),static_cast<long long>(k)*n,b,ptr(2),out_type,std::max(1,m),static_cast<long long>(m)*n,batch,compute,-1);
    } else if(r.op==1) {
      int order=v(0),m=v(1),n=v(2),batch=v(3),width=v(4);session.create(true);
      float af=float(r.scalars.at(0));double ad=r.scalars.at(0);
      const void* alpha=width==8?static_cast<const void*>(&ad):static_cast<const void*>(&af);
      for(int i=0;i<batch;++i)r.call(width==8?"cublasDtrsm_v2":"cublasStrsm_v2",session.handle,1-v(5),v(6),v(7),v(8),m,n,
        alpha,ptr(0,uint64_t(i)*order*order*width),std::max(1,order),ptr(1,uint64_t(i)*m*n*width),std::max(1,m));
    } else if(r.op==2) {
      int m=v(0),n=v(1),batch=v(2),width=v(3),k=std::min(m,n),length=0;session.create(false);
      r.call(prefix+"getrf_bufferSize",session.handle,m,n,ptr(0),std::max(1,m),&length);auto work=session.work(length,width);
      for(int i=0;i<batch;++i)r.call(prefix+"getrf",session.handle,m,n,ptr(0,uint64_t(i)*m*n*width),std::max(1,m),work,
        v(4)?ptr(1,uint64_t(i)*k*4):nullptr,ptr(2,uint64_t(i)*4));
    } else if(r.op==3) {
      int n=v(0),rhs=v(1),batch=v(2),width=v(3);session.create(false);
      for(int i=0;i<batch;++i)r.call(prefix+"getrs",session.handle,v(4),n,rhs,ptr(0,uint64_t(i)*n*n*width),std::max(1,n),
        ptr(1,uint64_t(i)*n*4),ptr(2,uint64_t(i)*n*rhs*width),std::max(1,n),ptr(3,uint64_t(i)*4));
    } else if(r.op==4) {
      int n=v(0),batch=v(1),width=v(2),upper=v(3),length=0;session.create(false);
      r.call(prefix+"potrf_bufferSize",session.handle,upper,n,ptr(0),std::max(1,n),&length);auto work=session.work(length,width);
      for(int i=0;i<batch;++i)r.call(prefix+"potrf",session.handle,upper,n,ptr(0,uint64_t(i)*n*n*width),std::max(1,n),work,length,ptr(1,uint64_t(i)*4));
    } else if(r.op==5) {
      int n=v(0),rhs=v(1),batch=v(2),width=v(3);session.create(false);
      for(int i=0;i<batch;++i)r.call(prefix+"potrs",session.handle,v(4),n,rhs,ptr(0,uint64_t(i)*n*n*width),std::max(1,n),
        ptr(1,uint64_t(i)*n*rhs*width),std::max(1,n),ptr(2,uint64_t(i)*4));
    } else if(r.op==6) {
      int m=v(0),n=v(1),batch=v(2),width=v(3),k=std::min(m,n),length=0;session.create(false);
      r.call(prefix+"geqrf_bufferSize",session.handle,m,n,ptr(0),std::max(1,m),&length);auto work=session.work(length,width);
      for(int i=0;i<batch;++i)r.call(prefix+"geqrf",session.handle,m,n,ptr(0,uint64_t(i)*m*n*width),std::max(1,m),ptr(1,uint64_t(i)*k*width),work,length,ptr(2,uint64_t(i)*4));
    } else if(r.op==7) {
      int m=v(0),n=v(1),k=v(2),batch=v(3),width=v(4),length=0;session.create(false);
      r.call(prefix+"orgqr_bufferSize",session.handle,m,n,k,ptr(0),std::max(1,m),ptr(1),&length);auto work=session.work(length,width);
      for(int i=0;i<batch;++i)r.call(prefix+"orgqr",session.handle,m,n,k,ptr(0,uint64_t(i)*m*n*width),std::max(1,m),ptr(1,uint64_t(i)*k*width),work,length,ptr(2,uint64_t(i)*4));
    } else if(r.op==8) {
      int order=v(0),columns=v(1),m=v(2),n=v(3),k=v(4),batch=v(5),width=v(6),trans=v(7),length=0;session.create(false);
      r.call(prefix+"ormqr_bufferSize",session.handle,0,trans,m,n,k,ptr(0),std::max(1,order),ptr(1),ptr(2),std::max(1,m),&length);auto work=session.work(length,width);
      for(int i=0;i<batch;++i)r.call(prefix+"ormqr",session.handle,0,trans,m,n,k,ptr(0,uint64_t(i)*order*columns*width),std::max(1,order),
        ptr(1,uint64_t(i)*k*width),ptr(2,uint64_t(i)*m*n*width),std::max(1,m),work,length,ptr(3,uint64_t(i)*4));
    } else if(r.op==9) {
      int n=v(0),batch=v(1),width=v(2),upper=v(3),vectors=v(4),length=0;session.create(false);
      r.call(prefix+"syevd_bufferSize",session.handle,vectors,upper,n,ptr(0),std::max(1,n),ptr(1),&length);auto work=session.work(length,width);
      for(int i=0;i<batch;++i)r.call(prefix+"syevd",session.handle,vectors,upper,n,ptr(0,uint64_t(i)*n*n*width),std::max(1,n),ptr(1,uint64_t(i)*n*width),work,length,ptr(2,uint64_t(i)*4));
    } else if(r.op==10 || r.op==11) {
      int m=v(0),n=v(1),uc=v(2),vc=v(3),batch=v(4),width=v(5),full=v(6),uv=v(7),k=std::min(m,n),length=0;session.create(false);
      if(r.op==10) {
        void* params=nullptr;r.call("cusolverDnCreateGesvdjInfo",&params);
        try {
          r.call(prefix+"gesvdj_bufferSize",session.handle,uv,1-full,m,n,ptr(0),std::max(1,m),ptr(1),ptr(2),std::max(1,m),ptr(3),std::max(1,n),&length,params);
          auto work=session.work(length,width);
          for(int i=0;i<batch;++i)r.call(prefix+"gesvdj",session.handle,uv,1-full,m,n,ptr(0,uint64_t(i)*m*n*width),std::max(1,m),ptr(1,uint64_t(i)*k*width),
            ptr(2,uint64_t(i)*m*uc*width),std::max(1,m),ptr(3,uint64_t(i)*n*vc*width),std::max(1,n),work,length,ptr(4,uint64_t(i)*4),params);
          r.call("cuStreamSynchronize",session.stream);r.call("cusolverDnDestroyGesvdjInfo",params);params=nullptr;
        } catch(...) {if(params)try {r.call("cuStreamSynchronize",session.stream);r.call("cusolverDnDestroyGesvdjInfo",params);}catch(...){} throw;}
      } else {
        r.call(prefix+"gesvd_bufferSize",session.handle,m,n,&length);auto work=session.work(length,width);auto rwork=session.work(std::max(1,k-1),width);
        signed char job=uv?(full?'A':'S'):'N';
        for(int i=0;i<batch;++i)r.call(prefix+"gesvd",session.handle,job,job,m,n,ptr(0,uint64_t(i)*m*n*width),std::max(1,m),ptr(1,uint64_t(i)*k*width),
          ptr(2,uint64_t(i)*m*uc*width),std::max(1,m),ptr(3,uint64_t(i)*vc*n*width),std::max(1,vc),work,length,rwork,ptr(4,uint64_t(i)*4));
      }
    } else if(r.op==12) {
      long long length=r.ints.at(0),batch=r.ints.at(1);int dbl=v(2),kind=v(3),plan=0;size_t bytes=0;
      if(dbl>2 || kind>3 || length<=0 || batch<=0)throw std::runtime_error("invalid cuFFT plan geometry/type");
      if(dbl==2 && (length&(length-1)))throw std::runtime_error("half cuFFT requires power-of-two length");
      long long in=kind==3?length/2+1:length,out=kind==2?length/2+1:length;
      int type=dbl?(kind==2?0x6a:kind==3?0x6c:0x69):(kind==2?0x2a:kind==3?0x2c:0x29);
      r.call("cufftCreate",&plan);
      try {
        r.call("cufftSetAutoAllocation",plan,0);r.call("cufftSetStream",plan,session.stream);
        if(dbl==2)r.call("cufftXtMakePlanMany",plan,1,&length,&in,static_cast<long long>(1),in,kind==2?2:6,
          &out,static_cast<long long>(1),out,kind==3?2:6,batch,&bytes,6);
        else r.call("cufftMakePlanMany64",plan,1,&length,&in,static_cast<long long>(1),in,&out,static_cast<long long>(1),out,type,batch,&bytes);
        auto workspace=bytes?session.alloc(bytes):nullptr;r.call("cufftSetWorkArea",plan,workspace);
        r.call("cuStreamSynchronize",session.stream);
        r.results={uint64_t(plan),reinterpret_cast<uintptr_t>(workspace),uint64_t(bytes)};
        if(workspace)session.allocations.pop_back();
      } catch(...) {try {r.call("cufftDestroy",plan);}catch(...){} throw;}
    } else if(r.op==13) {
      int plan=v(0),kind=v(1),dbl=v(2);r.call("cufftSetStream",plan,session.stream);
      if(dbl==2)r.call("cufftXtExec",plan,ptr(0),ptr(1),(kind==0 || kind==2)?-1:1);
      else if(kind==0 || kind==1)r.call(dbl?"cufftExecZ2Z":"cufftExecC2C",plan,ptr(0),ptr(1),kind==0?-1:1);
      else r.call(dbl?(kind==2?"cufftExecD2Z":"cufftExecZ2D"):(kind==2?"cufftExecR2C":"cufftExecC2R"),plan,ptr(0),ptr(1));
    } else if(r.op==14) {
      r.call("cufftDestroy",v(0));if(r.ints.at(1))r.call("cuMemFree_v2",uint64_t(r.ints.at(1)));
    } else if(r.op==15) {
      int n=v(0),incx=v(1),incy=v(2),type=v(3),conjugate=v(4);session.create(true);
      r.call("cublasSetPointerMode_v2",session.handle,1);
      if(type==2 || type==14) {
        r.call("cublasDotEx",session.handle,n,ptr(0),type,incx,ptr(1),type,incy,ptr(2),type,0);
      } else {
        const char* symbol=type==0?"cublasSdot_v2":type==1?"cublasDdot_v2":
          type==4?(conjugate?"cublasCdotc_v2":"cublasCdotu_v2"):
          type==5?(conjugate?"cublasZdotc_v2":"cublasZdotu_v2"):nullptr;
        if(!symbol)throw std::runtime_error("unsupported cuBLAS dot dtype");
        r.call(symbol,session.handle,n,ptr(0),incx,ptr(1),incy,ptr(2));
      }
    } else if(r.op==16) {
      const int m=v(1),n=v(0),k=v(2);
      session.destroy="cublasLtDestroy";r.call("cublasLtCreate",&session.handle);
      void* operation=nullptr;void* a_layout=nullptr;void* b_layout=nullptr;void* c_layout=nullptr;
      auto release=[&](const char* name,void*& descriptor) noexcept {
        if(descriptor) {
          void* current=descriptor;descriptor=nullptr;
          try {r.call(name,current);} catch(const std::exception& error) {
            r.error+=(r.error.empty()?"":"; ")+std::string("cuBLASLt descriptor cleanup: ")+error.what();
          }
        }
      };
      auto cleanup=[&]() noexcept {
        release("cublasLtMatrixLayoutDestroy",c_layout);release("cublasLtMatrixLayoutDestroy",b_layout);
        release("cublasLtMatrixLayoutDestroy",a_layout);release("cublasLtMatmulDescDestroy",operation);
      };
      try {
        r.call("cublasLtMatmulDescCreate",&operation,72,10);
        r.call("cublasLtMatrixLayoutCreate",&a_layout,3,uint64_t(m),uint64_t(k),int64_t(m));
        r.call("cublasLtMatrixLayoutCreate",&b_layout,3,uint64_t(k),uint64_t(n),int64_t(k));
        r.call("cublasLtMatrixLayoutCreate",&c_layout,10,uint64_t(m),uint64_t(n),int64_t(m));
        int32_t alpha=1,beta=0;void* nil=nullptr;
        r.call("cublasLtMatmul",session.handle,operation,&alpha,ptr(1),a_layout,ptr(0),b_layout,&beta,
               ptr(2),c_layout,ptr(2),c_layout,nil,nil,size_t(0),session.stream);
        r.call("cuStreamSynchronize",session.stream);
      } catch(...) {cleanup();throw;}
      cleanup();
    } else if(r.op==17) {
      const int m=v(1),n=v(0),k=v(2),a_type=v(4),b_type=v(3),output_type=v(5),bias_type=v(6);
      const int modes[]={0,3,4,5,4,5,2,1};
      if(v(7)>7 || v(8)>7)throw std::runtime_error("unsupported cuBLASLt scale mode");
      const bool device_alpha=r.ints.size()>11 && v(11)!=0;
      session.destroy="cublasLtDestroy";r.call("cublasLtCreate",&session.handle);
      void* operation=nullptr;void* a_layout=nullptr;void* b_layout=nullptr;
      void* c_layout=nullptr;void* d_layout=nullptr;void* preference=nullptr;
      auto release=[&](const char* name,void*& descriptor) noexcept {
        if(descriptor) {
          void* current=descriptor;descriptor=nullptr;
          try {r.call(name,current);} catch(const std::exception& error) {
            r.error+=(r.error.empty()?"":"; ")+std::string("cuBLASLt descriptor cleanup: ")+error.what();
          }
        }
      };
      auto cleanup=[&]() noexcept {
        release("cublasLtMatmulPreferenceDestroy",preference);release("cublasLtMatrixLayoutDestroy",d_layout);
        release("cublasLtMatrixLayoutDestroy",c_layout);release("cublasLtMatrixLayoutDestroy",b_layout);
        release("cublasLtMatrixLayoutDestroy",a_layout);release("cublasLtMatmulDescDestroy",operation);
      };
      try {
        r.call("cublasLtMatmulDescCreate",&operation,68,0);
        auto attribute=[&](int name,const auto& value) {r.call("cublasLtMatmulDescSetAttribute",operation,name,&value,sizeof(value));};
        const int trans_a=1,trans_b=0;attribute(3,trans_a);attribute(4,trans_b);
        if(device_alpha) {const int pointer_mode=1;attribute(2,pointer_mode);}
        void* scale_a=ptr(3);void* scale_b=ptr(2);attribute(17,scale_a);attribute(18,scale_b);
        const int8_t fast=int8_t(v(9));attribute(25,fast);
        if(v(7)!=0 || v(8)!=0) {attribute(31,modes[v(8)]);attribute(32,modes[v(7)]);}
        r.call("cublasLtMatrixLayoutCreate",&a_layout,a_type,uint64_t(k),uint64_t(m),int64_t(k));
        r.call("cublasLtMatrixLayoutCreate",&b_layout,b_type,uint64_t(k),uint64_t(n),int64_t(k));
        r.call("cublasLtMatrixLayoutCreate",&c_layout,bias_type,uint64_t(m),uint64_t(n),int64_t(m));
        r.call("cublasLtMatrixLayoutCreate",&d_layout,output_type,uint64_t(m),uint64_t(n),int64_t(m));
        if(v(10)) {
          void* bias=ptr(5);const int epilogue=4;
          attribute(8,bias);attribute(7,epilogue);attribute(26,bias_type);
        }
        r.call("cublasLtMatmulPreferenceCreate",&preference);
        const size_t limit=size_t(32)*1024*1024;
        r.call("cublasLtMatmulPreferenceSetAttribute",preference,1,&limit,sizeof(limit));
        struct Algorithm {uint64_t data[8];};
        struct Heuristic {Algorithm algo;size_t workspaceSize;int state;float wavesCount;int reserved[4];};
        Heuristic heuristic{};int returned=0;
        r.call("cublasLtMatmulAlgoGetHeuristic",session.handle,operation,a_layout,b_layout,c_layout,d_layout,preference,1,&heuristic,&returned);
        if(returned==0 || heuristic.state!=0)throw std::runtime_error("cuBLASLt has no scaled_mm algorithm for this device/layout/dtype");
        if(heuristic.workspaceSize>limit)throw std::runtime_error("cuBLASLt returned a workspace exceeding its preference");
        void* work=heuristic.workspaceSize?session.alloc(heuristic.workspaceSize):nullptr;
        const float alpha=1.0f,beta=0.0f;
        const void* alpha_pointer=device_alpha?ptr(5+v(10)):static_cast<const void*>(&alpha);
        const void* beta_pointer=device_alpha?ptr(6+v(10)):static_cast<const void*>(&beta);
        r.call("cublasLtMatmul",session.handle,operation,alpha_pointer,ptr(1),a_layout,ptr(0),b_layout,beta_pointer,
               ptr(1),c_layout,ptr(4),d_layout,&heuristic.algo,work,heuristic.workspaceSize,session.stream);
        r.call("cuStreamSynchronize",session.stream);
      } catch(...) {cleanup();throw;}
      cleanup();
    } else throw std::runtime_error("unknown vendor operation");
    session.finish();
    return r.error.empty()?0:-1;
  } catch(const std::exception& error) {r.error+=(r.error.empty()?"":"; ")+std::string(error.what());return -1;}
    catch(...) {r.error+="unknown vendor exception";return -1;}
}

std::vector<uint64_t> vendor_execute(uint32_t op,const std::vector<at::Tensor>& tensors,
    const std::vector<int64_t>& ints,const std::vector<double>& scalars,
    const std::unordered_map<std::string,uintptr_t>& functions) {
  TORCH_CHECK(vendor_scope_native,"RUDA vendor API unavailable; rebuild Rust and C++ libraries");
  std::vector<Argument> args;args.reserve(tensors.size());for(const auto& tensor:tensors)args.emplace_back(tensor);
  std::vector<Descriptor> descriptors;descriptors.reserve(args.size());for(const auto& arg:args)descriptors.push_back(arg.desc);
  VendorRequest request{op,ints,scalars,functions,{},{}};
  uint64_t stream=stream_command(0);int result;
  {
    // Pure native callback: no GIL acquisition on the device service thread.
    pybind11::gil_scoped_release release_gil;
    result=vendor_scope_native(vendor_callback,reinterpret_cast<uintptr_t>(&request),stream,descriptors.data(),descriptors.size());
  }
  TORCH_CHECK(result==0,request.error.empty()?(error_native?error_native():"vendor scope failed"):request.error);
  return request.results;
}
