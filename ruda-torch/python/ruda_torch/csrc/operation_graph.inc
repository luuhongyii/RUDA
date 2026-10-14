using OperationGraphCommand=int (*)(uint32_t,void**,uint32_t,const Descriptor*,size_t,
    const int64_t*,size_t,const float*,size_t,uint64_t,uint64_t,uint64_t*);
OperationGraphCommand operation_graph_native=nullptr;

class OperationGraphBridge {
  struct Binding {
    void* allocation;void* address;int64_t offset;c10::ScalarType dtype;
    std::vector<int64_t> shape,strides;
  };
  void* plan_=nullptr;
  std::vector<at::Tensor> tensors_;
  std::vector<Binding> bindings_;
  uint64_t stream_=0,kernels_=0;
  size_t inputs_=0;
  uint32_t kind_=0;
  std::mutex mutex_;
  uint64_t command(uint32_t op,uint64_t seed=0,uint64_t counter=0) {
    uint64_t result=0;
    check(operation_graph_native(op,&plan_,kind_,nullptr,0,nullptr,0,nullptr,0,seed,counter,&result));
    return result;
  }
  void check_bindings() const {
    TORCH_CHECK(plan_ && stream_command(0)==stream_,"operation graph is closed or its queue changed");
    for(size_t i=0;i<tensors_.size();++i) {
      const auto& t=tensors_[i];const auto& b=bindings_[i];validate(t.device());
      TORCH_CHECK(!t.requires_grad() && !t.is_conj() && !t.is_neg(),"operation graph bindings need detached resolved tensors");
      TORCH_CHECK(t.storage().data_ptr().get_deleter()==release && t.storage().data_ptr().get_context()==b.allocation
        && t.data_ptr()==b.address && t.storage_offset()==b.offset && t.scalar_type()==b.dtype
        && t.sizes()==c10::IntArrayRef(b.shape) && t.strides()==c10::IntArrayRef(b.strides),"operation graph binding changed");
      if(i>=inputs_)TORCH_CHECK(!t.is_inference() || c10::InferenceMode::is_enabled(),"operation graph writes inference storage outside inference mode");
    }
  }
 public:
  OperationGraphBridge(uint32_t kind,const std::vector<at::Tensor>& tensors,const std::vector<int64_t>& integers,
      const std::vector<float>& scalars,const std::vector<int64_t>& score_shape):tensors_(tensors),kind_(kind) {
    TORCH_CHECK(operation_graph_native && kind<=26,"operation graph API 1 unavailable or invalid kind");
    const size_t counts[]={6,10,6,9,5,3,6,5,7,3,9,15,8,3,2,2},inputs[]={4,6,4,6,4,2,5,4,2,1,6,9,6,2,1,1};
    TORCH_CHECK((kind==26 ? tensors.size()>=2 && tensors.size()<=3 : kind==25 ? tensors.size()>=3 && tensors.size()<=4 : kind>=23 ? tensors.size()==(kind==23 ? 4 : 6) : kind>=21 ? tensors.size()==(kind==21 ? 3 : 4) : kind==19 ? tensors.size()==2 : kind>=17 ? tensors.size()>=2 && tensors.size()<=3 : kind==16 ? tensors.size()>=2 : tensors.size()==counts[kind]) && integers.size()<=(kind==19 ? 19 : 8) && scalars.size()<=4,"invalid operation graph signature");
    inputs_=kind==24 ? 4 : (kind==20 || kind==26) ? 1 : kind>=16 ? tensors.size()-1 : inputs[kind];stream_=stream_command(0);
    for(size_t i=0;i<tensors_.size();++i) {
      const auto& t=tensors_[i];
      validate(t.device());
      TORCH_CHECK(t.layout()==c10::Layout::Strided && (t.is_contiguous() || (kind==26 && i>=inputs_ && t.is_non_overlapping_and_dense())) && !t.requires_grad()
        && !t.is_conj() && !t.is_neg(),"operation graph requires contiguous detached tensors");
      TORCH_CHECK(t.numel()<=std::numeric_limits<uint32_t>::max(),"operation graph indexing overflow");
      bindings_.push_back({t.storage().data_ptr().get_context(),t.data_ptr(),t.storage_offset(),t.scalar_type(),t.sizes().vec(),t.strides().vec()});
    }
    for(size_t i=inputs_;i<tensors_.size();++i) {
      at::assert_no_internal_overlap(tensors_[i]);
      for(size_t j=0;j<i;++j)at::assert_no_overlap(tensors_[i],tensors_[j]);
    }
    std::vector<Argument> args;args.reserve(tensors.size());
    for(size_t i=0;i<tensors.size();++i) {
      if(kind<=1 && i==3) {
        TORCH_CHECK(integers.size()==3,"invalid dense graph policy");
        args.emplace_back(integers[0]!=0 ? tensors[i].expand(score_shape) : tensors[i]);
      } else args.emplace_back(tensors[i]);
    }
    std::vector<Descriptor> descriptors;descriptors.reserve(args.size());
    for(const auto& arg:args)descriptors.push_back(arg.desc);
    check(operation_graph_native(0,&plan_,kind,descriptors.data(),descriptors.size(),integers.data(),integers.size(),
      scalars.data(),scalars.size(),0,0,&kernels_));
    TORCH_CHECK(plan_,"operation graph returned a null handle");
  }
  ~OperationGraphBridge() {
    if(plan_) {
      uint64_t ignored=0;
      if(operation_graph_native(4,&plan_,kind_,nullptr,0,nullptr,0,nullptr,0,0,0,&ignored))
        std::fprintf(stderr,"RUDA operation graph cleanup failed (native resources retained): %s\n",error_native());
    }
  }
  void run(uint64_t seed=0,uint64_t counter=0) {
    std::lock_guard<std::mutex> guard(mutex_);check_bindings();
    if(kind_==1 || kind_==3)at::globalContext().alertNotDeterministic("ruda::compiled_attention_backward");
    for(size_t i=inputs_;i<tensors_.size();++i)tensors_[i].unsafeGetTensorImpl()->bump_version();
    command(1,seed,counter);
  }
  void synchronize(){std::lock_guard<std::mutex> guard(mutex_);TORCH_CHECK(plan_,"operation graph closed");command(2);}
  bool query(){std::lock_guard<std::mutex> guard(mutex_);TORCH_CHECK(plan_,"operation graph closed");return command(3)!=0;}
  void close(){std::lock_guard<std::mutex> guard(mutex_);if(plan_)command(4);tensors_.clear();bindings_.clear();}
  uint64_t kernel_count()const{return kernels_;}
};
