using FftExecute=int (*)(uint32_t,const Descriptor*,size_t,const int64_t*,size_t,const double*,size_t);
using FftPlanExecute=int (*)(uint32_t,void**,const Descriptor*,size_t,const int64_t*,size_t,uint64_t*);
FftExecute fft_native=nullptr;
FftPlanExecute fft_plan_native=nullptr;

void fft_bridge_call(uint32_t op,const std::vector<at::Tensor>& tensors,
    const std::vector<int64_t>& ints,const std::vector<double>& scalars) {
  TORCH_CHECK(fft_native,"RUDA FFT API 1 unavailable; rebuild both native libraries");
  TORCH_CHECK(!tensors.empty(),"FFT command requires tensors");
  std::vector<Argument> args;args.reserve(tensors.size());
  for(const auto& t:tensors)args.emplace_back(t);
  at::assert_no_internal_overlap(tensors.back());
  std::vector<Descriptor> descriptors;descriptors.reserve(args.size());
  for(const auto& a:args)descriptors.push_back(a.desc);
  check(fft_native(op,descriptors.data(),descriptors.size(),ints.data(),ints.size(),scalars.data(),scalars.size()));
}

class FftPlanBridge {
  void* handle_=nullptr;
  std::mutex mutex_;
 public:
  FftPlanBridge(const at::Tensor& input,const std::vector<int64_t>& spec) {
    TORCH_CHECK(fft_plan_native,"RUDA FFT plan API unavailable");
    Argument arg(input);
    check(fft_plan_native(0,&handle_,&arg.desc,1,spec.data(),spec.size(),nullptr));
  }
  FftPlanBridge(const FftPlanBridge&)=delete;
  FftPlanBridge& operator=(const FftPlanBridge&)=delete;
  ~FftPlanBridge() {
    if(handle_ && fft_plan_native && fft_plan_native(4,&handle_,nullptr,0,nullptr,0,nullptr))
      std::fprintf(stderr,"RUDA FFT plan release: %s\n",error_native?error_native():"native error");
  }
  void execute(const at::Tensor& input,const at::Tensor& output) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(handle_,"FFT plan has been closed");
    at::assert_no_internal_overlap(output);
    Argument a(input),b(output);Descriptor ds[]={a.desc,b.desc};
    check(fft_plan_native(1,&handle_,ds,2,nullptr,0,nullptr));
  }
  uint64_t retained_bytes() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(handle_,"FFT plan has been closed");
    uint64_t bytes=0;check(fft_plan_native(3,&handle_,nullptr,0,nullptr,0,&bytes));return bytes;
  }
  void clear_workspace() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(handle_,"FFT plan has been closed");
    check(fft_plan_native(2,&handle_,nullptr,0,nullptr,0,nullptr));
  }
  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(handle_)check(fft_plan_native(4,&handle_,nullptr,0,nullptr,0,nullptr));
  }
};
