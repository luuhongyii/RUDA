using QwenGraphCommand=int (*)(uint32_t,void**,const Descriptor*,size_t,const uint64_t*,size_t,
    const float*,size_t,uint64_t,uint64_t*);
using QwenFlashCommand=int (*)(const Descriptor*,size_t,bool,float);
QwenGraphCommand qwen_graph_native=nullptr;
QwenFlashCommand qwen_flash_native=nullptr;

void qwen_flash(const std::vector<at::Tensor>& tensors,bool causal,double scale) {
  TORCH_CHECK(qwen_flash_native && tensors.size()==4,"Qwen Tensor Core attention unavailable");
  std::vector<Argument> args;args.reserve(4);
  for(const auto& t:tensors){training_tensor(t);args.emplace_back(t);}
  at::assert_no_internal_overlap(tensors[3]);
  for(size_t i=0;i<3;++i)at::assert_no_overlap(tensors[3],tensors[i]);
  std::vector<Descriptor> descriptors;for(const auto& a:args)descriptors.push_back(a.desc);
  check(qwen_flash_native(descriptors.data(),descriptors.size(),causal,float(scale)));
}

class QwenGraphBridge {
  void* plan_=nullptr;
  std::vector<at::Tensor> tensors_;
  std::vector<void*> addresses_;
  uint64_t stream_=0,kernels_=0;
  std::mutex mutex_;
  void command(uint32_t op,const at::Tensor* input=nullptr,uint64_t length=0) {
    uint64_t result=0;
    if(input){Argument arg(*input);check(qwen_graph_native(op,&plan_,&arg.desc,1,nullptr,0,nullptr,0,length,&result));}
    else check(qwen_graph_native(op,&plan_,nullptr,0,nullptr,0,nullptr,0,length,&result));
  }
 public:
  QwenGraphBridge(const std::vector<at::Tensor>& tensors,const std::vector<uint64_t>& dims,
      const std::vector<float>& scalars,uint64_t length):tensors_(tensors) {
    TORCH_CHECK(qwen_graph_native && dims.size()==7 && scalars.size()==2 && tensors.size()==4+9*dims[0],"invalid Qwen graph signature");
    stream_=stream_command(0);
    std::vector<Argument> args;args.reserve(tensors.size());
    for(const auto& t:tensors){training_tensor(t);TORCH_CHECK(!t.requires_grad(),"Qwen graph needs detached inference bindings");
      args.emplace_back(t);addresses_.push_back(t.data_ptr());}
    for(size_t i=0;i<tensors.size();++i){
      bool writable=i==3 || (i>=4 && (i-4)%9>=7);
      if(writable){at::assert_no_internal_overlap(tensors[i]);
        for(size_t j=0;j<tensors.size();++j)if(i!=j)at::assert_no_overlap(tensors[i],tensors[j]);}
    }
    std::vector<Descriptor> descriptors;for(const auto& a:args)descriptors.push_back(a.desc);
    check(qwen_graph_native(0,&plan_,descriptors.data(),descriptors.size(),dims.data(),dims.size(),
      scalars.data(),scalars.size(),length,&kernels_));
  }
  ~QwenGraphBridge(){if(plan_){uint64_t ignored=0;if(qwen_graph_native(3,&plan_,nullptr,0,nullptr,0,nullptr,0,0,&ignored))
      std::fprintf(stderr,"RUDA Qwen graph cleanup failed: %s\n",error_native());}}
  void run(const at::Tensor& hidden,uint64_t length,bool eager=false){
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(plan_ && stream_command(0)==stream_,"Qwen graph closed or stream changed");training_tensor(hidden);
    for(size_t i=0;i<tensors_.size();++i)TORCH_CHECK(tensors_[i].data_ptr()==addresses_[i],"Qwen graph binding changed");
    for(size_t i=0;i<tensors_.size();++i)if(i==3 || (i>=4 && (i-4)%9>=7))tensors_[i].unsafeGetTensorImpl()->bump_version();
    command(eager?4:1,&hidden,length);
  }
  void close(){std::lock_guard<std::mutex> guard(mutex_);if(plan_)command(3);tensors_.clear();addresses_.clear();}
  void synchronize(){std::lock_guard<std::mutex> guard(mutex_);TORCH_CHECK(plan_,"Qwen graph closed");command(2);}
  uint64_t kernel_count()const{return kernels_;}
};
