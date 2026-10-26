#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <ATen/EmptyTensor.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <c10/util/accumulate.h>
#include <ATen/native/MaxPooling.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/Padding.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/std.h>
#include <torch/custom_class.h>
#include <ATen/ops/_choose_qparams_per_tensor_native.h>
#include <ATen/ops/quantize_per_tensor_dynamic_native.h>
#include <ATen/core/List.h>
#include <ATen/native/TensorShape.h>
#include <ATen/ops/prelu.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/quantize_per_channel_native.h>
#include <ATen/ops/dequantize_native.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/_empty_per_channel_affine_quantized_native.h>
#include <ATen/ops/empty_quantized_native.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor_native.h>
#include <ATen/ops/_make_per_channel_quantized_tensor_native.h>
#include <ATen/ops/int_repr_native.h>
#include <ATen/ops/q_scale_native.h>
#include <ATen/ops/q_zero_point_native.h>
#include <ATen/ops/q_per_channel_scales_native.h>
#include <ATen/ops/q_per_channel_zero_points_native.h>
#include <ATen/ops/q_per_channel_axis_native.h>
#include <ATen/ops/qscheme_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ops/empty_strided.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/complex.h>
#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/Resize.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/ForeachUtils.h>
#if __has_include(<ATen/ops/_foreach_clone_native.h>)
#include <ATen/ops/_foreach_clone_native.h>
#endif
#if __has_include(<ATen/ops/_foreach_powsum_native.h>)
#include <ATen/ops/_foreach_powsum_native.h>
#endif
#if __has_include(<ATen/ops/linalg__powsum_native.h>)
#include <ATen/ops/linalg__powsum_native.h>
#endif
#include <ATen/native/BinaryOps.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/angle_native.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/deg2rad_native.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac_native.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/nan_to_num_native.h>
#include <ATen/ops/rad2deg_native.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/trunc_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/_conj_physical_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/ccol_indices_native.h>
#include <ATen/ops/row_indices_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/_to_sparse_native.h>
#include <ATen/ops/_to_sparse_csr_native.h>
#include <ATen/ops/_to_sparse_csc_native.h>
#include <ATen/ops/_to_sparse_bsr_native.h>
#include <ATen/ops/_to_sparse_bsc_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/sparse_resize_native.h>
#include <ATen/ops/sparse_mask_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/sparse_resize_and_clear_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/copy_sparse_to_sparse_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros_native.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/_sparse_broadcast_to_native.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/narrow_copy_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/norm_native.h>
#include <ATen/ops/native_norm_native.h>
#include <ATen/native/Math.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <torch/csrc/utils/pybind.h> // Tensor caster without the full C++ frontend
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/library.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <c10/core/DeviceGuard.h>
#include <cstdio>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <memory>
#include <mutex>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <c10/core/InferenceMode.h>
#include <c10/core/GradMode.h>

namespace {
struct Descriptor {
  void* allocation;
  size_t offset_bytes;
  size_t rank;
  const size_t* shape;
  const size_t* strides;
  uint32_t dtype;
};
using Alloc = int (*)(size_t, void**, uint64_t*);
using Free = int (*)(void*);
using Error = const char* (*)();
using Execute = int (*)(uint32_t, const Descriptor*, const Descriptor*, const Descriptor*, float);
using Addmm = int (*)(const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, float, float);
using LayerNorm = int (*)(const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, float);
using RMSNorm = int (*)(const Descriptor*, const Descriptor*, const Descriptor*, float);
using Spatial = int (*)(uint32_t, const Descriptor*, const Descriptor*, const Descriptor*, const int64_t*, size_t);
using Fill = int (*)(const Descriptor*, uint64_t);
using Transfer = int (*)(const Descriptor*, void*, bool);
using Sync = int (*)();
using StreamInterop = int (*)(uint32_t,uint64_t,uint64_t,uint32_t,uint64_t*);
using Paged = int (*)(uint32_t,void**,const Descriptor*,const Descriptor*,const Descriptor*,
                     const Descriptor*,const Descriptor*,const Descriptor*,
                     const Descriptor*,const Descriptor*,const Descriptor*,const Descriptor*,const Descriptor*,const Descriptor*,
                     const uint32_t*,size_t,const uint32_t*,float,bool);
Alloc allocate_native = nullptr;
Free free_native = nullptr;
Error error_native = nullptr;
Execute execute_native = nullptr;
Addmm addmm_native = nullptr;
LayerNorm layer_norm_native = nullptr;
RMSNorm rms_norm_native = nullptr;
Spatial spatial_native = nullptr;
Fill fill_native = nullptr;
Transfer transfer_native = nullptr;
Sync sync_native = nullptr;
StreamInterop stream_native = nullptr;
Paged paged_native = nullptr;
bool paged_backward_selected_ready = false;
uint32_t paged_backward_api = 0;
using Scientific = int (*)(uint32_t,const Descriptor*,size_t,const int64_t*,size_t,const double*,size_t);
Scientific scientific_native = nullptr;

void check(int code) { TORCH_CHECK(code == 0, error_native ? error_native() : "RUDA bridge is not initialized"); }
void synchronize() { TORCH_CHECK(sync_native, "RUDA bridge is not initialized"); check(sync_native()); }
void validate(c10::Device device) {
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 && device.index() <= 0,
              "RUDA currently exposes only ruda:0");
}
bool fp8_storage_available=false;
bool mxfp8_storage_available=false;
bool fp4_storage_available=false;
uint32_t dtype_code(c10::ScalarType dtype) {
  if(dtype==at::ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK(fp4_storage_available,"packed FP4 storage requires matching rebuilt FP4 API 1 libraries");
    return 16;
  }
  if(dtype==at::ScalarType::Float8_e8m0fnu) {
    TORCH_CHECK(mxfp8_storage_available,"E8M0 storage requires matching rebuilt MXFP8 API 1 libraries");
    return 15;
  }
  if (dtype == at::kFloat) return 0;
  if (dtype == at::kHalf) return 1;
  if (dtype == at::kBFloat16) return 2;
  if (dtype == at::kBool) return 3;
  if (dtype == at::kLong) return 4;
  if (dtype == at::kInt) return 5;
  if (dtype == at::kShort) return 6;
  if (dtype == at::kChar) return 7;
  if (dtype == at::kByte) return 8;
  if (dtype == at::ScalarType::Float8_e4m3fn || dtype == at::ScalarType::Float8_e5m2) {
    TORCH_CHECK(fp8_storage_available,"FP8 storage requires matching packed API 1 native libraries");
    return dtype == at::ScalarType::Float8_e4m3fn ? 12 : 13;
  }
  if (dtype == at::kDouble || dtype == at::kComplexFloat || dtype == at::kComplexDouble || dtype == at::kComplexHalf) {
    TORCH_CHECK(scientific_native,"RUDA scientific storage API 1 requires matching rebuilt native libraries");
    return dtype==at::kDouble ? 9 : dtype==at::kComplexFloat ? 10 : dtype==at::kComplexDouble ? 11 : 14;
  }
  TORCH_CHECK(false, "RUDA native storage supports float32, float16, bfloat16, bool, int64, int32, int16, int8 and uint8");
}
void release(void* handle) {
  if (handle && free_native(handle)) std::fprintf(stderr, "RUDA release: %s\n", error_native());
}
class Allocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    TORCH_CHECK(allocate_native, "RUDA bridge is not initialized");
    void* handle = nullptr;
    uint64_t ptr = 0;
    check(allocate_native(bytes, &handle, &ptr));
    return c10::DataPtr(reinterpret_cast<void*>(ptr), handle, release,
                       c10::Device(c10::DeviceType::PrivateUse1, 0));
  }
  void copy_data(void*, const void*, size_t) const override {
    TORCH_CHECK(false, "RUDA raw storage copy is not implemented; use Tensor.copy_");
  }
};
Allocator allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);

#include "storage.inc"
#include "generator.inc"

uint64_t stream_command(uint32_t op, uint64_t stream=0, uint64_t object=0, uint32_t flags=0) {
  TORCH_CHECK(stream_native, "RUDA ABI 8 stream bridge is not initialized");
  uint64_t result=0;check(stream_native(op,stream,object,flags,&result));return result;
}
c10::Stream make_stream(uint64_t id) {
  return c10::Stream(c10::Stream::UNSAFE,c10::Device(c10::DeviceType::PrivateUse1,0),id);
}
void check_stream(const c10::Stream& stream) { validate(stream.device()); }
class Guard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }
  c10::Device getDevice() const override { return c10::Device(type(),0); }
  void setDevice(c10::Device device) const override {validate(device);}
  c10::Device exchangeDevice(c10::Device device) const override {validate(device);return getDevice();}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device device) const override {validate(device);return make_stream(stream_command(0));}
  c10::Stream getDefaultStream(c10::Device device) const override {validate(device);return make_stream(0);}
  c10::Stream getNewStream(c10::Device device,int priority=0) const override {
    validate(device);
    // Native ABI 10 previously ignored op 1 flags. The additive query must
    // succeed before any nonzero priority request reaches that old command.
    if(priority!=0)stream_command(15);
    return make_stream(stream_command(1,0,0,static_cast<uint32_t>(priority)));
  }
  c10::Stream getStreamFromGlobalPool(c10::Device device,bool high=false) const override {
    return getNewStream(device,high ? -1 : 0);
  }
  c10::Stream exchangeStream(c10::Stream value) const override {check_stream(value);return make_stream(stream_command(2,value.id()));}
  c10::DeviceIndex deviceCount() const noexcept override {return 1;}
  bool queryStream(const c10::Stream& value) const override {check_stream(value);return stream_command(3,value.id())!=0;}
  void synchronizeStream(const c10::Stream& value) const override {check_stream(value);stream_command(4,value.id());}
  void synchronizeDevice(c10::DeviceIndex index) const override {validate(c10::Device(type(),index));stream_command(11);}
  void record(void** event,const c10::Stream& stream,c10::DeviceIndex index,c10::EventFlag flag) const override {
    validate(c10::Device(type(),index));check_stream(stream);
    TORCH_CHECK(flag!=c10::EventFlag::INVALID,"invalid event flag");
    *event=reinterpret_cast<void*>(stream_command(5,stream.id(),reinterpret_cast<uintptr_t>(*event),flag==c10::EventFlag::BACKEND_DEFAULT));
  }
  void block(void* event,const c10::Stream& stream) const override {check_stream(stream);stream_command(6,stream.id(),reinterpret_cast<uintptr_t>(event));}
  bool queryEvent(void* event) const override {return stream_command(7,0,reinterpret_cast<uintptr_t>(event))!=0;}
  void synchronizeEvent(void* event) const override {stream_command(8,0,reinterpret_cast<uintptr_t>(event));}
  void destroyEvent(void* event,c10::DeviceIndex) const noexcept override {
    if (!event || !stream_native) return;
    uint64_t ignored=0;
    if(stream_native(9,0,reinterpret_cast<uintptr_t>(event),0,&ignored))
      std::fprintf(stderr,"RUDA event destruction: %s\n",error_native());
  }
  double elapsedTime(void* a,void* b,c10::DeviceIndex index) const override {
    validate(c10::Device(type(),index));uint32_t bits=stream_command(10,reinterpret_cast<uintptr_t>(a),reinterpret_cast<uintptr_t>(b));
    float ms;std::memcpy(&ms,&bits,sizeof(ms));return ms;
  }
  void recordDataPtrOnStream(const c10::DataPtr& ptr,const c10::Stream& stream) const override {
    check_stream(stream);TORCH_CHECK(ptr.get_deleter()==release,"record_stream requires RUDA storage");
    if(ptr.get_context())stream_command(12,stream.id(),reinterpret_cast<uintptr_t>(ptr.get_context()));
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, Guard);

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  const at::Generator& getDefaultGenerator(c10::DeviceIndex index) const override {
    validate(c10::Device(c10::DeviceType::PrivateUse1,index)); return default_generator();
  }
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t bytes) const override {
    resize_storage(storage, bytes);
  }
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return allocate_native != nullptr; }
  bool hasPrimaryContext(c10::DeviceIndex device) const override { return device == 0 && isAvailable(); }
  c10::DeviceIndex deviceCount() const override { return 1; }
  void setCurrentDevice(c10::DeviceIndex device) const override { validate(c10::Device(c10::DeviceType::PrivateUse1, device)); }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override { setCurrentDevice(device); return 0; }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override { return exchangeDevice(device); }
};

struct Argument {
  std::vector<size_t> shape, strides;
  Descriptor desc;
  explicit Argument(const at::Tensor& tensor,bool physical=false) {
    validate(tensor.device());
    TORCH_CHECK(physical || (!tensor.is_conj() && !tensor.is_neg()), "resolve RUDA conjugate/negative view bits before numerical dispatch");
    TORCH_CHECK(tensor.storage().data_ptr().get_deleter() == release, "tensor is not allocated by RUDA");
    for (auto d : tensor.sizes()) { TORCH_CHECK(d >= 0); shape.push_back(d); }
    for (auto d : tensor.strides()) { TORCH_CHECK(d >= 0); strides.push_back(d); }
    desc = {tensor.storage().data_ptr().get_context(), tensor.numel() ? size_t(tensor.storage_offset()) * tensor.element_size() : 0,
            shape.size(), shape.data(), strides.data(), dtype_code(tensor.scalar_type())};
  }
};

#include "scientific_bridge.inc"
#include "blas_bridge.inc"
#include "linalg_bridge.inc"
#include "fft_bridge.inc"
#include "vendor_bridge.inc"

#include "indexing.inc"
#include "training.inc"
#include "router.inc"
#include "nf4.inc"
#include "sequence_training.inc"
#include "kda.inc"
#include "packed.inc"
#include "random.inc"
#include "dense_attention.inc"
#include "varlen_attention.inc"
#include "ring_attention.inc"
#include "float8.inc"
#include "sorting.inc"
#include "ordering.inc"
#include "statistics.inc"

class PagedPlanBridge {
  void* plan_=nullptr;
 public:
  at::Tensor run_quantized(const at::Tensor& q,const at::Tensor& qp,
      const std::vector<at::Tensor>& caches,const std::vector<uint32_t>& layout,double scale,bool causal) {
    TORCH_CHECK(quantized_paged_native && caches.size()==9 && layout.size()==10 && q.dim()==3,
                "quantized paged API 1 unavailable or invalid arguments");
    training_tensor(q); training_tensor(qp);
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(float(scale)) && scale>0,"invalid attention scale");
    auto out=at::empty({q.size(0),q.size(1),int64_t(layout[6])},q.options());
    std::vector<Argument> args;args.reserve(12);args.emplace_back(q);args.emplace_back(qp);
    for (const auto& t:caches) {
      TORCH_CHECK(t.device()==q.device() && t.is_contiguous(),"quantized cache device/layout mismatch");
      args.emplace_back(t);
    }
    args.emplace_back(out);
    std::vector<Descriptor> ds;ds.reserve(12);for (const auto& a:args)ds.push_back(a.desc);
    check(quantized_paged_native(plan_,ds.data(),ds.size(),layout.data(),layout.size(),float(scale),causal));
    return out;
  }
  PagedPlanBridge(const at::Tensor& like,const std::vector<uint32_t>& spec,const std::vector<uint32_t>& words) {
    TORCH_CHECK(paged_native && spec.size()==6,"invalid native paged plan header");
    Argument q(like);
    check(paged_native(0,&plan_,&q.desc,nullptr,nullptr,nullptr,nullptr,nullptr,
                       nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
                       words.data(),words.size(),spec.data(),1.0f,true));
  }
  PagedPlanBridge(const PagedPlanBridge&)=delete;
  PagedPlanBridge& operator=(const PagedPlanBridge&)=delete;
  ~PagedPlanBridge() {
    if(plan_ && paged_native(2,&plan_,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr,1.0f,true))
      std::fprintf(stderr,"RUDA paged plan release: %s\n",error_native());
  }
  at::Tensor run(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                 const std::optional<at::Tensor>& qp,const std::optional<at::Tensor>& kp,
                 double scale,bool causal) {
    TORCH_CHECK(q.dim()==3 && k.dim()==4 && v.dim()==4,"paged Q/cache ranks must be 3/4/4");
    TORCH_CHECK(qp.has_value()==kp.has_value(),"MLA requires both positional tensors");
    auto out=at::empty({q.size(0),q.size(1),v.size(3)},q.options());
    for(const auto& t: {q,k,v})at::assert_no_overlap(out,t);
    Argument aq(q),ak(k),av(v),ao(out);
    auto ap=qp?std::make_unique<Argument>(*qp):nullptr;
    auto akp=kp?std::make_unique<Argument>(*kp):nullptr;
    check(paged_native(1,&plan_,&aq.desc,&ak.desc,&av.desc,ap?&ap->desc:nullptr,akp?&akp->desc:nullptr,
                       &ao.desc,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
                       nullptr,0,nullptr,static_cast<float>(scale),causal));
    return out;
  }
  std::vector<at::Tensor> backward(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                 const std::optional<at::Tensor>& qp,const std::optional<at::Tensor>& kp,
                 const at::Tensor& grad,double scale,bool causal) {
    TORCH_CHECK(qp.has_value()==kp.has_value(),"MLA requires both positional tensors");
    TORCH_CHECK(grad.sizes()==at::IntArrayRef({q.size(0),q.size(1),v.size(3)}),
                "paged backward gradient shape mismatch");
    if (q.numel()!=0) at::globalContext().alertNotDeterministic("ruda::paged_history_backward");
    auto dq=at::empty_like(q); auto dk=at::empty_like(k);
    at::assert_no_overlap(dq,q);at::assert_no_overlap(dk,k);
    Argument aq(q),ak(k),av(v),ag(grad),adq(dq),adk(dk);
    auto ap=qp?std::make_unique<Argument>(*qp):nullptr;
    auto akp=kp?std::make_unique<Argument>(*kp):nullptr;
    if(!qp) {
      auto dv=at::empty_like(v);at::assert_no_overlap(dv,v);Argument adv(dv);
      check(paged_native(3,&plan_,&aq.desc,&ak.desc,&av.desc,nullptr,nullptr,nullptr,
                         &ag.desc,&adq.desc,&adk.desc,&adv.desc,nullptr,nullptr,
                         nullptr,0,nullptr,static_cast<float>(scale),causal));
      return {dq,dk,dv};
    }
    auto dqp=at::empty_like(*qp);auto dkp=at::empty_like(*kp);
    at::assert_no_overlap(dqp,*qp);at::assert_no_overlap(dkp,*kp);
    Argument adqp(dqp),adkp(dkp);
    check(paged_native(3,&plan_,&aq.desc,&ak.desc,&av.desc,&ap->desc,&akp->desc,nullptr,
                       &ag.desc,&adq.desc,&adk.desc,nullptr,&adqp.desc,&adkp.desc,
                       nullptr,0,nullptr,static_cast<float>(scale),causal));
    return {dq,dqp,dk,dkp};
  }
  std::vector<std::optional<at::Tensor>> backward_selected(
      const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
      const std::optional<at::Tensor>& qp,const std::optional<at::Tensor>& kp,
      const at::Tensor& grad,double scale,bool causal,const std::vector<bool>& needs,bool ordered=false) {
    TORCH_CHECK(paged_backward_selected_ready,
                "RUDA paged backward API 1 unavailable; rebuild Rust and C++ extensions");
    TORCH_CHECK(!ordered || paged_backward_api>=2,"RUDA ordered history backward requires paged API 2; rebuild Rust and C++");
    TORCH_CHECK(qp.has_value()==kp.has_value(),"MLA requires both positional tensors");
    TORCH_CHECK(q.dim()==3 && k.dim()==4 && v.dim()==4,"paged backward ranks must be 3/4/4");
    const bool mla=qp.has_value();
    TORCH_CHECK(needs.size()==(mla?4:3),"invalid paged gradient request length");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"paged scale must be finite and positive");
    TORCH_CHECK(grad.sizes()==at::IntArrayRef({q.size(0),q.size(1),v.size(3)}),
                "paged backward gradient shape mismatch");
    std::vector<at::Tensor> inputs{q,k,v,grad};
    if(mla) { inputs.push_back(*qp); inputs.push_back(*kp); }
    for(const auto& t:inputs) {
      validate(t.device());
      TORCH_CHECK(t.is_contiguous() && t.scalar_type()==q.scalar_type(),
                  "paged backward operands must be contiguous and same dtype");
    }
    TORCH_CHECK(q.scalar_type()==at::kFloat || q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,
                "paged backward requires floating storage");
    // Determinism is a property of the requested derivative, not of requires_grad
    // on every input. Query-only derivatives have no shared-history atomics.
    const bool atomic=mla?(needs[2] || needs[3]):(needs[1] || needs[2]);
    if(atomic && !ordered && q.numel()!=0) at::globalContext().alertNotDeterministic("ruda::paged_history_backward");
    const std::vector<at::Tensor> sources=mla?std::vector<at::Tensor>{q,*qp,k,*kp}:std::vector<at::Tensor>{q,k,v};
    std::vector<std::optional<at::Tensor>> result(sources.size());
    std::vector<std::unique_ptr<Argument>> args(sources.size());
    for(size_t i=0;i<sources.size();++i) if(needs[i]) {
      result[i]=at::empty_like(sources[i]);
      for(const auto& input:inputs) at::assert_no_overlap(*result[i],input);
      for(size_t j=0;j<i;++j) if(result[j]) at::assert_no_overlap(*result[i],*result[j]);
      args[i]=std::make_unique<Argument>(*result[i]);
    }
    if(std::none_of(needs.begin(),needs.end(),[](bool value){return value;})) return result;
    Argument aq(q),ak(k),av(v),ag(grad);
    auto ap=qp?std::make_unique<Argument>(*qp):nullptr;
    auto akp=kp?std::make_unique<Argument>(*kp):nullptr;
    auto desc=[&](size_t i)->const Descriptor* {return args[i]?&args[i]->desc:nullptr;};
    check(paged_native(ordered?5:4,&plan_,&aq.desc,&ak.desc,&av.desc,ap?&ap->desc:nullptr,akp?&akp->desc:nullptr,nullptr,
        &ag.desc,desc(0),desc(mla?2:1),mla?nullptr:desc(2),mla?desc(1):nullptr,mla?desc(3):nullptr,
        nullptr,0,nullptr,static_cast<float>(scale),causal));
    return result;
  }

};

#include "static_graph.inc"
#include "operation_graph.inc"
#include "collective_program.inc"

at::Tensor empty(c10::IntArrayRef size, std::optional<c10::ScalarType> dtype,
  std::optional<c10::Layout> layout, std::optional<c10::Device> device,
  std::optional<bool> pin, std::optional<c10::MemoryFormat> format) {
  validate(c10::device_or_default(device));
  dtype_code(c10::dtype_or_default(dtype));
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided && !c10::pinned_memory_or_default(pin));
  return at::detail::empty_generic(size, &allocator, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), c10::dtype_or_default(dtype), format);
}
at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
  std::optional<c10::ScalarType> dtype, std::optional<c10::Layout> layout,
  std::optional<c10::Device> device, std::optional<bool> pin) {
  validate(c10::device_or_default(device));
  dtype_code(c10::dtype_or_default(dtype));
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided && !c10::pinned_memory_or_default(pin));
  return at::detail::empty_strided_generic(size, stride, &allocator,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), c10::dtype_or_default(dtype));
}
at::Tensor efficient_zero(c10::SymIntArrayRef size, std::optional<c10::ScalarType> dtype,
  std::optional<c10::Layout> layout, std::optional<c10::Device> device, std::optional<bool> pin) {
  auto target = c10::device_or_default(device);
  validate(target);
  if (!target.has_index()) target = c10::Device(c10::DeviceType::PrivateUse1, 0);
  dtype_code(c10::dtype_or_default(dtype));
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided && !c10::pinned_memory_or_default(pin));
  at::native::ZeroTensorAllocator zero_allocator(target);
  return at::detail::empty_generic_symint(size, &zero_allocator,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1) | c10::DispatchKeySet(c10::DispatchKey::ZeroTensor),
      c10::dtype_or_default(dtype), std::nullopt);
}
void execute(uint32_t op, const at::Tensor& a, const at::Tensor& b, at::Tensor out, float scalar) {
  at::assert_no_internal_overlap(out);
  if (op == 7 || op == 30) {
    at::assert_no_overlap(out, a);
    at::assert_no_overlap(out, b);
  } else {
    at::assert_no_partial_overlap(out, a);
    at::assert_no_partial_overlap(out, b);
  }
  Argument av(a), bv(b), ov(out);
  check(execute_native(op, &av.desc, &bv.desc, &ov.desc, scalar));
}

void addmm(const at::Tensor& bias, const at::Tensor& a, const at::Tensor& b,
           at::Tensor out, float alpha, float beta) {
  TORCH_CHECK(addmm_native, "RUDA ABI 5 is required; rebuild Rust and C++ extensions");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, a);
  at::assert_no_overlap(out, b);
  at::assert_no_overlap(out, bias);
  Argument cv(bias), av(a), bv(b), ov(out);
  check(addmm_native(&cv.desc, &av.desc, &bv.desc, &ov.desc, alpha, beta));
}

#include "matmul_dispatch.inc"

void layer_norm(const at::Tensor& input, const std::optional<at::Tensor>& weight,
                const std::optional<at::Tensor>& bias, at::Tensor out, at::Tensor mean,
                at::Tensor rstd, double epsilon) {
  TORCH_CHECK(layer_norm_native, "RUDA ABI 6 is required; rebuild Rust and C++ extensions");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, input);
  if (weight) at::assert_no_overlap(out, *weight);
  if (bias) at::assert_no_overlap(out, *bias);
  Argument iv(input), ov(out), mv(mean), rv(rstd);
  std::optional<Argument> wv, bv;
  if (weight) wv.emplace(*weight);
  if (bias) bv.emplace(*bias);
  check(layer_norm_native(&iv.desc, wv ? &wv->desc : nullptr, bv ? &bv->desc : nullptr,
                          &ov.desc, &mv.desc, &rv.desc, static_cast<float>(epsilon)));
}

void rms_norm(const at::Tensor& input, const std::optional<at::Tensor>& weight, at::Tensor out, double epsilon) {
  TORCH_CHECK(rms_norm_native, "RUDA ABI 7 is required; rebuild Rust and C++ extensions");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, input);
  if (weight) at::assert_no_overlap(out, *weight);
  Argument iv(input), ov(out);
  std::optional<Argument> wv;
  if (weight) wv.emplace(*weight);
  check(rms_norm_native(&iv.desc, wv ? &wv->desc : nullptr, &ov.desc, static_cast<float>(epsilon)));
}

void spatial(uint32_t op, const at::Tensor& a, const at::Tensor& b, at::Tensor out,
             const std::vector<int64_t>& params) {
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, a);
  at::assert_no_overlap(out, b);
  if (op == 6) {
    at::assert_no_internal_overlap(b);
    at::assert_no_overlap(b, a);
  }
  Argument av(a), bv(b), ov(out);
  check(spatial_native(op, &av.desc, &bv.desc, &ov.desc, params.data(), params.size()));
}

template <typename T>
uint64_t scalar_bits(const at::Scalar& value) {
  const T converted = value.to<T>();
  uint64_t bits = 0;
  std::memcpy(&bits, &converted, sizeof(T));
  return bits;
}
void fill(at::Tensor out, const at::Scalar& value) {
  at::assert_no_internal_overlap(out);
  if(out.is_conj() || out.is_neg()) {
    auto logical=at::empty(out.sizes(),out.options());
    (fill)(logical,value);
    scientific_call(207,{logical,out},{int64_t(out.is_conj()),int64_t(out.is_neg())},{});
    return;
  }
  Argument ov(out);
  uint64_t bits = 0;
  switch (ov.desc.dtype) {
    case 0: bits = scalar_bits<float>(value); break;
    case 1: bits = scalar_bits<at::Half>(value); break;
    case 2: bits = scalar_bits<at::BFloat16>(value); break;
    case 3: bits = scalar_bits<bool>(value); break;
    case 4: bits = scalar_bits<int64_t>(value); break;
    case 5: bits = scalar_bits<int32_t>(value); break;
    case 6: bits = scalar_bits<int16_t>(value); break;
    case 7: bits = scalar_bits<int8_t>(value); break;
    case 8: bits = scalar_bits<uint8_t>(value); break;
    case 9: bits = scalar_bits<double>(value); break;
    case 12: { c10::Float8_e4m3fn v(value.toFloat()); bits=v.x; break; }
    case 13: { c10::Float8_e5m2 v(value.toFloat()); bits=v.x; break; }
    case 15: { c10::Float8_e8m0fnu v(value.toFloat()); bits=v.x; break; }
    case 16: TORCH_CHECK(false,"scalar fill is not defined for packed Float4_e2m1fn_x2; use a packed byte view");
    case 10: {
      auto number=value.toComplexDouble();
      float pair[2]={float(number.real()),float(number.imag())};
      std::memcpy(&bits,pair,sizeof(bits));
      break;
    }
    case 11: {
      auto number=value.toComplexDouble();
      scientific_fill(out,number.real(),number.imag());
      return;
    }
    case 14: {
      auto number=value.toComplexDouble();
      at::Half pair[2]={at::Half(static_cast<float>(number.real())),at::Half(static_cast<float>(number.imag()))};
      std::memcpy(&bits,pair,sizeof(pair));
      break;
    }
  }
  check(fill_native(&ov.desc, bits));
}

at::Tensor quantized_copy_from(const at::Tensor& source, const at::Tensor& dest, bool non_blocking);

at::Tensor copy_from(const at::Tensor& source, const at::Tensor& dest, bool non_blocking) {
  if (source.is_quantized() || dest.is_quantized()) return quantized_copy_from(source, dest, non_blocking);
  TORCH_CHECK(!dest._is_zerotensor(), "ZeroTensors are immutable. Please materialize using .clone() before mutation.");
  if (source._is_zerotensor()) return const_cast<at::Tensor&>(dest).zero_();
  dtype_code(source.scalar_type());
  dtype_code(dest.scalar_type());
  if(source.scalar_type()==at::ScalarType::Float4_e2m1fn_x2 || dest.scalar_type()==at::ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK(source.scalar_type()==dest.scalar_type(),"packed FP4 copy does not perform numeric conversion");
  }
  at::assert_no_internal_overlap(dest);
  if(source.is_same(dest))return dest;
  if(!dest.device().is_cpu() && (dest.is_conj() || dest.is_neg())) {
    auto logical=at::empty(dest.sizes(),dest.options());
    copy_from(source,logical,non_blocking);
    scientific_call(207,{logical,dest},{int64_t(dest.is_conj()),int64_t(dest.is_neg())},{});
    return dest;
  }
  auto resolved=source.device().is_cpu()?source.resolve_conj().resolve_neg():scientific_resolve(source);
  auto input=resolved.expand(dest.sizes());
  if (dest.numel()!=0 && source.is_complex() &&
      (dest.scalar_type()==at::ScalarType::Float8_e4m3fn || dest.scalar_type()==at::ScalarType::Float8_e5m2 || dest.scalar_type()==at::ScalarType::Float8_e8m0fnu)) {
    TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
  }
  if (source.device().is_cpu()) {
    auto packed = input.contiguous();
    auto staging = source.scalar_type() == dest.scalar_type() ? dest : at::empty(dest.sizes(), dest.options().dtype(source.scalar_type()));
    Argument target(staging);
    check(transfer_native(&target.desc, packed.data_ptr(), true));
    if (!staging.is_same(dest)) execute(0, staging, staging, dest, 0);
  } else if (dest.device().is_cpu()) {
    auto staging = source.scalar_type() == dest.scalar_type() ? input : at::empty(dest.sizes(), source.options().dtype(dest.scalar_type()));
    if (!staging.is_same(input)) execute(0, input, input, staging, 0);
    auto packed = at::empty(dest.sizes(), dest.options());
    Argument origin(staging);
    check(transfer_native(&origin.desc, packed.data_ptr(), false));
    const_cast<at::Tensor&>(dest).copy_(packed, non_blocking);
  } else {
    if (source.is_same(dest)) return dest;
    execute(0, input, input, dest, 0);
  }
  return dest;
}
at::Tensor& copy_(at::Tensor& dest, const at::Tensor& source, bool non_blocking) { copy_from(source, dest, non_blocking); return dest; }
#include "factories.inc"
#include "rank_one.inc"
#include "random_extra.inc"
#include "fused_optim.inc"
#include "foreach_unary.inc"
#include "foreach_binary.inc"
#include "arithmetic_dispatch.inc"
#include "power.inc"
#include "foreach_ternary.inc"
#include "foreach_reduce.inc"
template <typename T>
T read_scalar(const at::Tensor& value) {
  T result{};
  Argument arg(value);
  check(transfer_native(&arg.desc, &result, false));
  return result;
}
at::Scalar scalar(const at::Tensor& value) {
  TORCH_CHECK(value.numel() == 1);
  if(value.is_conj() || value.is_neg())return scalar(scientific_resolve(value));
  switch (dtype_code(value.scalar_type())) {
    case 16: TORCH_CHECK(false,"item() is not defined for packed Float4_e2m1fn_x2");
    case 3: return read_scalar<uint8_t>(value) != 0;
    case 4: return read_scalar<int64_t>(value);
    case 5: return int64_t(read_scalar<int32_t>(value));
    case 6: return int64_t(read_scalar<int16_t>(value));
    case 7: return int64_t(read_scalar<int8_t>(value));
    case 8: return int64_t(read_scalar<uint8_t>(value));
    case 9: return read_scalar<double>(value);
    case 10: return read_scalar<c10::complex<float>>(value);
    case 11: return read_scalar<c10::complex<double>>(value);
    case 14: {
      auto number=read_scalar<c10::complex<at::Half>>(value);
      return c10::complex<double>(static_cast<float>(number.real()),static_cast<float>(number.imag()));
    }
  }
  auto converted = value.scalar_type() == at::kFloat ? value : value.to(at::kFloat);
  return read_scalar<float>(converted);
}
void unsupported(const c10::OperatorHandle& op, torch::jit::Stack*) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "RUDA GPU operator not implemented: ", op.schema().operator_name(), "; CPU fallback is disabled");
}
#include "fused_autograd_dispatch.inc"
void record_tensor_stream(at::Tensor& value, c10::Stream stream) {
  check_stream(stream); Argument arg(value);
  stream_command(12,stream.id(),reinterpret_cast<uintptr_t>(arg.desc.allocation));
}
TORCH_LIBRARY_IMPL(ruda, PrivateUse1, m) {
  m.impl("lora_epilogue_forward",lora_epilogue_forward);
  m.impl("lora_epilogue_backward",lora_epilogue_backward);
  m.impl("lora_rank16_residual",lora_rank16_residual);
  m.impl("training_rms_forward",training_rms_forward_impl);
  m.impl("training_rms_backward",training_rms_backward_op);
  m.impl("training_layer_forward",training_layer_forward);
  m.impl("training_layer_backward",training_layer_backward_op);
  m.impl("training_silu_forward",training_silu_forward);
  m.impl("training_silu_backward",training_silu_backward_op);
  m.impl("rotary_qk_forward",rotary_qk_forward);
  m.impl("rotary_qk_backward",rotary_qk_backward);
  m.impl("cross_entropy_forward",cross_entropy_forward);
  m.impl("cross_entropy_backward",cross_entropy_backward);
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", native_add_sub_dispatch<0>);
  m.impl("add_.Tensor", native_add_sub_inplace_dispatch<0>);
  m.impl("sub.Tensor", native_add_sub_dispatch<1>);
  m.impl("sub_.Tensor", native_add_sub_inplace_dispatch<1>);
  m.impl("mul.Tensor", native_mul_dispatch);
  m.impl("mul_.Tensor", native_mul_inplace_dispatch);
  m.impl("neg", native_neg_dispatch);
  m.impl("neg_", native_neg_inplace_dispatch);
  m.impl("mm", native_mm_dispatch);
  m.impl("bmm", native_bmm_dispatch);
  m.impl("addmm", native_addmm_dispatch);
  m.impl("addr", TORCH_FN(at::native::addr));
  m.impl("addr.out", TORCH_FN(at::native::addr_out));
  m.impl("topk",topk_tensor);
  m.impl("topk.values",topk_tensor_out);
  m.impl("sort",sort_tensor);
  m.impl("sort.stable",sort_stable_tensor);
  m.impl("sort.values",sort_tensor_out);
  m.impl("sort.values_stable",sort_stable_tensor_out);
  m.impl("argsort",argsort_tensor);
  m.impl("argsort.stable",argsort_stable_tensor);
  m.impl("argsort.stable_out",argsort_stable_tensor_out);
  m.impl("index.Tensor",advanced_index);
  m.impl("index.Tensor_out",advanced_index_out);
  m.impl("set_.source_Storage", set_storage);
  m.impl("set_.source_Storage_storage_offset", set_storage_offset);
  m.impl("resize_", resize_tensor);
  m.impl("normal_", normal_);
  m.impl("arange.start_out", arange_out);
  m.impl("range.out", range_out);
  m.impl("linspace.out", linspace_out);
  m.impl("logspace.out", logspace_out);
  m.impl("record_stream",record_tensor_stream);
  m.impl("empty.memory_format", empty);
  m.impl("_efficientzerotensor", efficient_zero);
  m.impl("empty_strided", empty_strided);
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("as_strided_", at::native::as_strided__symint);
  m.impl("view", at::native::view);
  m.impl("view.dtype", [](const at::Tensor& self, c10::ScalarType dtype) {
    dtype_code(dtype);
    return at::native::view_dtype(self,dtype);
  });
  m.impl("_reshape_alias", at::native::_reshape_alias);
  m.impl("copy_", copy_);
  m.impl("_copy_from", copy_from);
  m.impl("_foreach_copy_", foreach_copy_);
#if __has_include(<ATen/ops/_foreach_clone_native.h>)
  m.impl("_foreach_clone", foreach_clone);
#endif
  m.impl("_local_scalar_dense", scalar);
  m.impl("view_as_real",at::native::view_as_real);
  m.impl("view_as_complex",at::native::view_as_complex);
}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&unsupported>());
}
#include "sparse_bridge.inc"
#include "quantized_bridge.inc"
#include "quantized_activation.inc"
#include "quantized_arithmetic.inc"
#include "quantized_extended.inc"
#include "quantized_normalization.inc"
#include "quantized_pooling.inc"
#include "quantized_linear.inc"
#include "quantized_convolution.inc"
#include "quantized_resize.inc"
#include "quantized_selection.inc"
#include "quantized_reduce.inc"
#include "quantized_index.inc"
#include "quantized_assignment.inc"
#include "quantized_embedding.inc"
// AMP for the compute-heavy dense projections. The wrapper redispatches below
// AutocastPrivateUse1, so the existing RUDA kernels remain the execution path.
// Normalization is intentionally not force-cast here: RUDA's native training
// kernels already keep their statistics/accumulation in FP32 while allowing
// low-precision activation storage.
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(bmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(linear, lower_precision_fp)
}
}

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(kaiser_window_stub, &::kaiser_window_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(addr_stub, &::addr_kernel_ruda);
REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_tensor_affine_stub, &::quantize_scalar_ruda);
REGISTER_PRIVATEUSE1_DISPATCH(dequantize_tensor_per_tensor_affine_stub, &::dequantize_scalar_ruda);
REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_channel_affine_stub, &::quantize_channel_ruda);
REGISTER_PRIVATEUSE1_DISPATCH(dequantize_tensor_per_channel_affine_stub, &::dequantize_channel_ruda);
REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_channel_float_qparams_stub, &::quantize_float_channel_ruda);
REGISTER_PRIVATEUSE1_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub, &::dequantize_float_channel_ruda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize_fused_autograd_dispatch", initialize_fused_autograd_dispatch);
  m.def("initialize_arithmetic_dispatch", initialize_arithmetic_dispatch);
  m.def("initialize_matmul_dispatch", initialize_matmul_dispatch);
  #include "scientific_bridge_bindings.inc"
  #include "blas_bridge_bindings.inc"
  #include "linalg_bridge_bindings.inc"
  #include "fft_bridge_bindings.inc"
  #include "vendor_bridge_bindings.inc"
  #include "dense_attention_bindings.inc"
  #include "varlen_attention_bindings.inc"
  #include "ring_attention_bindings.inc"
  #include "collective_program_bindings.inc"
  #include "float8_bindings.inc"
  m.attr("abi_version") = 10;
  m.def("_cuda_alias", [](const at::Tensor& tensor, int64_t device_index) {
    validate(tensor.device());
    TORCH_CHECK(tensor.layout() == c10::Layout::Strided && tensor.is_contiguous(), "NCCL alias requires contiguous RUDA storage");
    TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(), "NCCL alias cannot use unresolved views");
    TORCH_CHECK(tensor.storage().data_ptr().get_deleter() == release, "NCCL alias requires RUDA allocation ownership");
    TORCH_CHECK(device_index >= 0 && device_index <= std::numeric_limits<c10::DeviceIndex>::max(), "invalid CUDA ordinal");
    // Alias, not a transfer: keep RUDA storage alive until NCCL has completed.
    return at::from_blob(tensor.data_ptr(), tensor.sizes(), tensor.strides(),
      [owner=tensor](void*) {}, tensor.options().device(c10::Device(c10::DeviceType::CUDA, device_index)).requires_grad(false));
  });
  m.attr("indexing_api_version") = 1;
  m.def("initialize_indexing",[](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !advanced_index_native,"invalid or repeated indexing initialization");
    advanced_index_native=reinterpret_cast<AdvancedIndex>(address);
  });
  m.attr("factory_api_version") = 1;
  m.def("initialize_factories", [](std::vector<uintptr_t> addresses) {
    TORCH_CHECK(allocate_native && addresses.size()==2 && addresses[0] && addresses[1] && !arange_native,
                "invalid or repeated factory initialization");
    arange_native = reinterpret_cast<Arange>(addresses[0]);
    normal_native = reinterpret_cast<Normal>(addresses[1]);
  });
  m.def("default_generator", [] { return default_generator(); });
  m.attr("nf4_matmul_api_version") = 1;
  m.def("initialize_nf4_matmul", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !nf4_matmul_native, "invalid or repeated NF4 matmul initialization");
    nf4_matmul_native=reinterpret_cast<NF4Matmul>(address);
  });
  m.def("nf4_matmul", nf4_matmul);
  m.attr("random_api_version") = 1;
  m.def("initialize_random",[](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !random_fill_native,"invalid or repeated random initialization");
    random_fill_native=reinterpret_cast<RandomFill>(address);
  });
  m.def("random_fill_",random_fill);
  m.def("random_bounds",random_bounds);
  m.def("uniform_parameters",uniform_parameters);
  m.def("random_launch_seed",random_launch_seed);
  m.def("randperm_prepare",randperm_prepare);
  m.def("fused_optimizer_check",fused_optimizer_check);
  m.def("fused_optimizer_cpu_lr",fused_optimizer_cpu_lr);
  m.def("foreach_unary_fast",foreach_unary_fast);
  m.def("foreach_unary_bump",foreach_unary_bump);
  m.def("foreach_unary_slow",foreach_unary_slow);
#if __has_include(<ATen/ops/linalg__powsum_native.h>)
  m.def("powsum_slow", [](const at::Tensor& self, const at::Scalar& ord,
                          const std::optional<std::vector<int64_t>>& dims, bool keepdim,
                          std::optional<at::ScalarType> dtype) {
    if (dims) return at::native::linalg__powsum_slow(self, ord, at::IntArrayRef(*dims), keepdim, dtype);
    return at::native::linalg__powsum_slow(self, ord, std::nullopt, keepdim, dtype);
  });
#endif
  m.attr("sequence_api_version") = 1;
  m.def("initialize_sequence", [](std::vector<uintptr_t> addresses) {
    TORCH_CHECK(allocate_native && addresses.size()==2 && addresses[0] && addresses[1] && !triangular_native,
                "invalid or repeated sequence initialization");
    triangular_native=reinterpret_cast<TriangularSolve>(addresses[0]);
    delta_native=reinterpret_cast<DeltaForward>(addresses[1]);
  });
  m.def("triangular_solve", triangular_solve);
  m.def("delta_forward", delta_forward);
  m.attr("kda_api_version") = 1;
  m.def("initialize_kda", [](std::vector<uintptr_t> addresses) {
    TORCH_CHECK(allocate_native && addresses.size()==2 && addresses[0] && addresses[1] && !kda_forward_native,
                "invalid or repeated KDA initialization");
    kda_forward_native=reinterpret_cast<KdaForward>(addresses[0]);
    kda_backward_native=reinterpret_cast<KdaBackward>(addresses[1]);
  });
  m.def("kda_forward", kda_forward);
  m.def("kda_backward", kda_backward);
  m.attr("packed_api_version") = 1;
  m.attr("mxfp8_api_version") = 1;
  m.attr("fp4_api_version") = 1;
  m.def("initialize_fp4", [](uint32_t version) {
    TORCH_CHECK(version==1 && allocate_native && scientific_native && mxfp8_storage_available && !fp4_storage_available,
                "invalid or repeated FP4 initialization");
    fp4_storage_available=true;
  });
  m.def("initialize_mxfp8", [](uint32_t version) {
    TORCH_CHECK(version==1 && allocate_native && scientific_native && fp8_storage_available && !mxfp8_storage_available,
                "invalid or repeated MXFP8 initialization");
    mxfp8_storage_available=true;
  });
  m.attr("packed_matmul_api_version") = 1;
  m.attr("packed_indexed_groups_api_version") = 1;
  m.def("initialize_packed", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !packed_native,"invalid or repeated packed initialization");
    packed_native=reinterpret_cast<PackedDispatch>(address);
    fp8_storage_available=true;
  });
  m.def("packed_dispatch", packed_dispatch);
  m.attr("paged_quantized_api_version") = 1;
  m.def("initialize_paged_quantized", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !quantized_paged_native,"invalid quantized paged initialization");
    quantized_paged_native=reinterpret_cast<QuantizedPaged>(address);
  });
  m.def("initialize", [](std::vector<uintptr_t> addresses) {
    TORCH_CHECK(addresses.size() == 13 && !allocate_native, "invalid or repeated RUDA initialization");
    for (auto address : addresses) TORCH_CHECK(address != 0, "null RUDA ABI function");
    allocate_native = reinterpret_cast<Alloc>(addresses[0]);
    free_native = reinterpret_cast<Free>(addresses[1]);
    error_native = reinterpret_cast<Error>(addresses[2]);
    execute_native = reinterpret_cast<Execute>(addresses[3]);
    transfer_native = reinterpret_cast<Transfer>(addresses[4]);
    sync_native = reinterpret_cast<Sync>(addresses[5]);
    fill_native = reinterpret_cast<Fill>(addresses[6]);
    spatial_native = reinterpret_cast<Spatial>(addresses[7]);
    addmm_native = reinterpret_cast<Addmm>(addresses[8]);
    layer_norm_native = reinterpret_cast<LayerNorm>(addresses[9]);
    rms_norm_native = reinterpret_cast<RMSNorm>(addresses[10]);
    stream_native = reinterpret_cast<StreamInterop>(addresses[11]);
    paged_native = reinterpret_cast<Paged>(addresses[12]);
    synchronize();
    at::RegisterPrivateUse1HooksInterface(new Hooks());
  });
  m.attr("paged_backward_api_version") = 2;
  m.def("initialize_paged_backward", [](uint32_t version) {
    TORCH_CHECK(allocate_native && (version==1 || version==2) && !paged_backward_selected_ready,
                "invalid or repeated paged backward initialization");
    paged_backward_selected_ready=true; paged_backward_api=version;
  });
  m.attr("router_api_version") = 1;
  m.attr("nf4_api_version") = 1;
  m.def("initialize_nf4", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !nf4_native, "invalid or repeated NF4 initialization");
    nf4_native = reinterpret_cast<NF4Decode>(address);
  });
  m.def("nf4_decode", nf4_decode);
  m.def("initialize_router", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !router_native, "invalid or repeated router initialization");
    router_native = reinterpret_cast<RouterCommand>(address);
  });
  m.def("router_weights_forward", router_weights_forward);
  m.def("router_weights_backward", router_weights_backward);
  m.attr("training_api_version") = 4;
  m.def("initialize_training", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !training_native, "invalid or repeated training initialization");
    training_native=reinterpret_cast<TrainingCommand>(address);
  });
  m.def("training_rms_forward", training_rms_forward);
  m.def("training_rms_backward", training_rms_backward);
  m.attr("training_fusion_api_version") = 3;
  m.def("training_hf_rms_forward", training_hf_rms_forward);
  m.def("training_hf_rms_backward", training_hf_rms_backward);
  m.def("training_layer_forward", training_layer_forward);
  m.def("training_layer_backward", training_layer_backward);
  m.def("training_silu_forward", training_silu_forward);
  m.def("training_silu_backward", training_silu_backward);
  m.def("training_unscale_", training_unscale);
  m.def("training_adamw_", training_adamw);
  m.def("training_analyze", training_analyze);
  m.def("training_analyze_hierarchical", training_analyze_hierarchical);
  m.def("training_adamw_batch_", training_adamw_batch);
  m.attr("graph_api_version") = 3;
  m.attr("graph_layout_api_version") = 1;
  m.attr("graph_math_api_version") = 1;
  m.attr("graph_index_api_version") = 1;
  m.attr("graph_factory_api_version") = 1;
  m.attr("graph_empty_api_version") = 1;
  m.attr("graph_root_api_version") = 1;
  m.attr("operation_graph_api_version") = 1;
  m.attr("sequence_graph_api_version") = 1;
  m.attr("pointwise_graph_api_version") = 1;
  m.attr("transform_graph_api_version") = 1;
  m.attr("reduction_graph_api_version") = 1;
  m.attr("router_graph_api_version") = 1;
  m.attr("grouped_graph_api_version") = 1;
  m.attr("scatter_graph_api_version") = 1;
  m.attr("sorting_graph_api_version") = 1;
  m.attr("block_float8_graph_api_version") = 1;
  m.attr("delayed_float8_graph_api_version") = 1;
  m.attr("sorting_api_version") = 1;
  m.attr("ordering_api_version") = 1;
  m.def("initialize_ordering",[](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !ordering_native,"invalid or repeated ordering initialization");
    ordering_native=reinterpret_cast<OrderingDispatch>(address);
  });
  m.def("ordering_call",ordering_call);
  m.attr("statistics_api_version") = 1;
  m.def("initialize_statistics",[](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !statistics_native,"invalid or repeated statistics initialization");
    statistics_native=reinterpret_cast<StatisticsDispatch>(address);
  });
  m.def("statistics_call",statistics_call);
  m.def("initialize_sorting",[](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !sorting_native,"invalid or repeated sorting initialization");
    sorting_native=reinterpret_cast<SortingDispatch>(address);
  });
  m.def("initialize_operation_graph",[](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !operation_graph_native,"invalid or repeated operation graph initialization");
    operation_graph_native=reinterpret_cast<OperationGraphCommand>(address);
  });
  pybind11::class_<OperationGraphBridge,std::shared_ptr<OperationGraphBridge>>(m,"NativeOperationGraph")
    .def(pybind11::init<uint32_t,const std::vector<at::Tensor>&,const std::vector<int64_t>&,const std::vector<float>&,const std::vector<int64_t>&>())
    .def("run",&OperationGraphBridge::run,pybind11::arg("seed")=0,pybind11::arg("counter")=0,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("synchronize",&OperationGraphBridge::synchronize,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("query",&OperationGraphBridge::query,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("close",&OperationGraphBridge::close,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def_property_readonly("kernel_count",&OperationGraphBridge::kernel_count);
  m.def("initialize_graph", [](uintptr_t address) {
    TORCH_CHECK(allocate_native && address && !static_graph_native, "invalid or repeated static graph initialization");
    static_graph_native = reinterpret_cast<StaticGraphCommand>(address);
  });
  pybind11::class_<StaticGraphBridge,std::shared_ptr<StaticGraphBridge>>(m,"NativeStaticGraph")
    .def(pybind11::init<const std::vector<at::Tensor>&,size_t,const std::vector<uint32_t>&,
         const std::vector<float>&,bool,bool>())
    .def("run",&StaticGraphBridge::run,pybind11::arg("eager")=false,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("synchronize",&StaticGraphBridge::synchronize,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("query",&StaticGraphBridge::query,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("query_completion",&StaticGraphBridge::query_completion,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("wait_completion",&StaticGraphBridge::wait_completion,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("close",&StaticGraphBridge::close,pybind11::call_guard<pybind11::gil_scoped_release>())
    .def_property_readonly("edge_count",&StaticGraphBridge::edge_count)
    .def_property_readonly("stream_id",&StaticGraphBridge::stream_id);
  m.def("stream_command",stream_command,pybind11::arg("op"),pybind11::arg("stream")=0,
        pybind11::arg("object")=0,pybind11::arg("flags")=0);
  m.def("record_stream",[](const at::Tensor& t,uint64_t id){Argument a(t);stream_command(12,id,reinterpret_cast<uintptr_t>(a.desc.allocation));});
  pybind11::class_<PagedPlanBridge,std::shared_ptr<PagedPlanBridge>>(m,"NativePagedPlan")
    .def(pybind11::init<const at::Tensor&,const std::vector<uint32_t>&,const std::vector<uint32_t>&>())
    .def("run",&PagedPlanBridge::run)
    .def("run_quantized",&PagedPlanBridge::run_quantized)
    .def("backward",&PagedPlanBridge::backward)
    .def("backward_selected",&PagedPlanBridge::backward_selected,
        pybind11::arg("q"),pybind11::arg("k"),pybind11::arg("v"),pybind11::arg("qp"),pybind11::arg("kp"),
        pybind11::arg("grad"),pybind11::arg("scale"),pybind11::arg("causal"),pybind11::arg("needs"),pybind11::arg("ordered")=false);
  m.def("execute", execute);
  m.def("addmm", addmm);
  m.def("layer_norm", layer_norm);
  m.def("rms_norm", rms_norm);
  m.def("spatial", spatial);
  m.def("extrema_result_type", [](const std::vector<at::Tensor>& inputs) {
    at::native::ResultTypeState state{};
    for (const auto& input : inputs) state = at::native::update_result_type_state(input, state);
    return at::native::result_type(state);
  });
  m.def("where_condition", [](const at::Tensor& condition) {
    if (condition.scalar_type() == at::kByte) {
      TORCH_WARN_ONCE("where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
      return condition.to(at::kBool);
    }
    TORCH_CHECK(condition.scalar_type() == at::kBool, "where expected condition to be a boolean tensor");
    return condition;
  });
  m.def("predicate_scalar", [](pybind11::object value, at::ScalarType dtype, at::Device device) {
    static torch::PythonArgParser parser({"predicate_scalar(Scalar value)"});
    torch::ParsedArgs<1> parsed;
    auto args = pybind11::make_tuple(value);
    const auto scalar_value = parser.parse(args.ptr(), nullptr, parsed).scalar(0);
    return at::native::wrapped_scalar_tensor(scalar_value).to(dtype).to(device);
  });
  m.def("arithmetic_alpha", [](pybind11::object value, at::ScalarType dtype, at::ScalarType output_dtype, bool subtract) {
    static torch::PythonArgParser parser({"arithmetic_alpha(Scalar value)"});
    torch::ParsedArgs<1> parsed;
    auto args = pybind11::make_tuple(value);
    const auto alpha = parser.parse(args.ptr(), nullptr, parsed).scalar(0);
    at::native::alpha_check(output_dtype, alpha);
    std::vector<int64_t> bits;
    foreach_binary_scalar_bits(bits, subtract ? -alpha : alpha, dtype);
    return bits[0];
  });
  m.def("compressed_from_dense", [](const at::Tensor& self, std::optional<at::Layout> layout,
      std::optional<std::vector<int64_t>> blocksize, std::optional<int64_t> dense_dim) {
    at::OptionalIntArrayRef blocks = std::nullopt;
    if (blocksize) blocks = at::IntArrayRef(*blocksize);
    return at::native::dense_to_sparse(self, layout, blocks, dense_dim);
  });
  m.def("compressed_flip_block", [](const at::Tensor& self, std::vector<int64_t> blocksize,
      std::optional<int64_t> dense_dim, bool column) {
    TORCH_CHECK(self.layout() == at::kSparseBsr || self.layout() == at::kSparseBsc,
        "compressed_flip_block requires blocked sparse storage");
    return column ? at::native::sparse_compressed_to_sparse_bsc(self, blocksize, dense_dim)
                  : at::native::sparse_compressed_to_sparse_bsr(self, blocksize, dense_dim);
  });
  m.def("compressed_mm_output", [](const at::Tensor& out, const at::Tensor& result) {
    TORCH_CHECK((result.layout() == at::kSparseCsr || result.layout() == at::kSparseCsc) &&
        out.layout() == result.layout() && out.scalar_type() == result.scalar_type() && out.device() == result.device(),
        "compressed matrix output must match result layout, dtype and device");
    TORCH_CHECK(out.dim() == 2 && result.dim() == 2 && out.dense_dim() == 0 && result.dense_dim() == 0,
        "compressed matrix output requires non-batched scalar sparse matrices");
    const bool row = result.layout() == at::kSparseCsr;
    static_cast<at::SparseCsrTensorImpl*>(out.unsafeGetTensorImpl())->set_member_tensors(
        row ? result.crow_indices() : result.ccol_indices(),
        row ? result.col_indices() : result.row_indices(), result.values(), result.sizes());
  });
  m.def("compressed_arithmetic_output", [](const at::Tensor& out, const at::Tensor& result) {
    const auto layout = result.layout();
    TORCH_CHECK((layout == at::kSparseCsr || layout == at::kSparseCsc || layout == at::kSparseBsr || layout == at::kSparseBsc) &&
        out.layout() == layout && out.scalar_type() == result.scalar_type() && out.device() == result.device(),
        "compressed arithmetic output must match result layout, dtype and device");
    const bool row = layout == at::kSparseCsr || layout == at::kSparseBsr;
    static_cast<at::SparseCsrTensorImpl*>(out.unsafeGetTensorImpl())->set_member_tensors(
        row ? result.crow_indices() : result.ccol_indices(),
        row ? result.col_indices() : result.row_indices(), result.values(), result.sizes());
  });
  m.def("sparse_alpha_is_one", [](pybind11::object value, at::ScalarType dtype) {
    static torch::PythonArgParser parser({"sparse_alpha_is_one(Scalar value)"});
    torch::ParsedArgs<1> parsed;
    auto args = pybind11::make_tuple(value);
    const auto alpha = parser.parse(args.ptr(), nullptr, parsed).scalar(0);
    bool one = false;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16, dtype, "sparse_alpha_is_one", [&] {
      one = alpha.to<scalar_t>() == scalar_t(1);
    });
    return one;
  });
  m.def("blas_accumulate_scalars", [](pybind11::object alpha_object, pybind11::object beta_object, at::ScalarType dtype, bool gemv) {
    static torch::PythonArgParser parser({"blas_accumulate_scalars(Scalar alpha, Scalar beta)"});
    torch::ParsedArgs<2> parsed;
    auto args = pybind11::make_tuple(alpha_object, beta_object);
    auto values = parser.parse(args.ptr(), nullptr, parsed);
    const auto alpha = values.scalar(0), beta = values.scalar(1);
    std::vector<double> coefficients(4, 0.0);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::kBFloat16, dtype, "blas_accumulate_scalars", [&] {
      using math_t = at::opmath_type<scalar_t>;
      const math_t a = gemv ? math_t(alpha.to<scalar_t>()) : alpha.to<math_t>();
      const math_t b = gemv ? math_t(beta.to<scalar_t>()) : beta.to<math_t>();
      auto pack = [&](auto av, auto bv) {
        if constexpr (c10::is_complex<decltype(av)>::value) {
          coefficients = {double(av.real()), double(av.imag()), double(bv.real()), double(bv.imag())};
        } else {
          coefficients[0] = double(av); coefficients[2] = double(bv);
        }
      };
      pack(a, b);
    });
    return pybind11::make_tuple(coefficients, beta.toComplexDouble() == c10::complex<double>(0.0, 0.0));
  });
  m.def("division_scalar", [](pybind11::object value, at::ScalarType dtype, int64_t op) {
    TORCH_CHECK(op >= 0 && op <= 2, "invalid scalar division operation");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kDouble,
                "scalar division requires a real floating computation dtype");
    at::Tensor scalar;
    if (THPVariable_Check(value.ptr())) {
      scalar = THPVariable_Unpack(value.ptr());
      TORCH_CHECK(scalar.device().is_cpu() && scalar.dim() == 0 && !scalar.is_complex(), "expected a real CPU scalar tensor");
      scalar = scalar.resolve_conj().resolve_neg();
    } else {
      static torch::PythonArgParser parser({"division_scalar(Scalar value)"});
      torch::ParsedArgs<1> parsed;
      auto args = pybind11::make_tuple(value);
      scalar = at::native::wrapped_scalar_tensor(parser.parse(args.ptr(), nullptr, parsed).scalar(0));
    }
    const double source = scalar.to(at::kDouble).const_data_ptr<double>()[0];
    auto pack = [](auto number) {
      uint64_t word = 0; std::memcpy(&word, &number, sizeof(number));
      int64_t bits; std::memcpy(&bits, &word, sizeof(bits)); return bits;
    };
    if (dtype == at::kDouble) return std::vector<int64_t>{pack(source), pack(1.0 / source)};
    const float divisor = scalar.to(at::kFloat).const_data_ptr<float>()[0];
    const float reciprocal = op == 0 || (op == 2 && divisor == 0)
        ? static_cast<float>(1.0 / source) : 1.0f / divisor;
    return std::vector<int64_t>{pack(divisor), pack(reciprocal)};
  });
  m.def("logit_epsilon", [](double value, at::ScalarType dtype) {
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kDouble,
                "logit requires a real floating computation dtype");
    return dtype == at::kDouble ? value : static_cast<double>(at::Scalar(value).to<float>());
  });
  m.def("power_scalar", power_scalar_parameters);
  m.def("power_base_log", power_scalar_base_log);
  m.def("masked_fill_native", [](const at::Tensor& input, const at::Tensor& mask, const at::Tensor& output,
                                 pybind11::object value, bool inplace) {
    validate(input.device());
    TORCH_CHECK(mask.device() == input.device() && output.device() == input.device(), "masked_fill device mismatch");
    TORCH_CHECK(mask.scalar_type() == at::kBool && input.scalar_type() == output.scalar_type(), "masked_fill dtype mismatch");
    TORCH_CHECK(input.sizes() == output.sizes(), "masked_fill output shape mismatch");
    if (inplace) {
      TORCH_CHECK(input.is_same(output), "in-place masked_fill must target the input tensor");
      if (at::has_internal_overlap(input) == at::MemOverlap::Yes) {
        TORCH_WARN("Use of masked_fill_ on expanded tensors is deprecated. Please clone() the tensor before performing this operation. This also applies to advanced indexing e.g. tensor[mask] = scalar");
      }
      at::assert_no_partial_overlap(input, mask);
    } else {
      at::assert_no_internal_overlap(output);
      TORCH_CHECK(!input.is_conj() && !input.is_neg() && !output.is_conj() && !output.is_neg(), "resolve functional masked_fill logical views before dispatch");
    }
    auto expanded_mask = mask.expand(input.sizes());
    static torch::PythonArgParser parser({"masked_fill_native(Scalar value)"});
    torch::ParsedArgs<1> parsed;
    auto args = pybind11::make_tuple(value);
    const auto scalar_value = parser.parse(args.ptr(), nullptr, parsed).scalar(0);
    uint64_t words[2] = {0, 0};
    const auto code = dtype_code(input.scalar_type());
    if (code == 12) {
      words[0] = c10::Float8_e4m3fn(scalar_value.toFloat()).x;
    } else if (code == 13) {
      words[0] = c10::Float8_e5m2(scalar_value.toFloat()).x;
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::kBool, at::kHalf, at::kBFloat16, at::kComplexHalf,
          input.scalar_type(), "ruda_masked_fill", [&] {
        const auto converted = scalar_value.to<scalar_t>();
        static_assert(sizeof(scalar_t) <= sizeof(words));
        std::memcpy(words, &converted, sizeof(scalar_t));
      });
    }
    if (output.is_neg()) {
      switch (code) {
        case 0: words[0] ^= 0x80000000ULL; break;
        case 1: case 2: words[0] ^= 0x8000ULL; break;
        case 4: case 5: case 6: case 7: case 8: words[0] = uint64_t(0) - words[0]; break;
        case 9: words[0] ^= 0x8000000000000000ULL; break;
        case 10: words[0] ^= 0x8000000080000000ULL; break;
        case 11: words[0] ^= 0x8000000000000000ULL; words[1] ^= 0x8000000000000000ULL; break;
        case 12: case 13: words[0] ^= 0x80ULL; break;
        case 14: words[0] ^= 0x80008000ULL; break;
        default: TORCH_CHECK(false, "masked_fill negative view dtype is unsupported");
      }
    }
    if (output.is_conj()) {
      if (code == 10) words[0] ^= 0x8000000000000000ULL;
      else if (code == 11) words[1] ^= 0x8000000000000000ULL;
      else if (code == 14) words[0] ^= 0x80000000ULL;
    }
    auto source = input;
    auto target = output;
    if (inplace && (input.is_conj() || input.is_neg())) {
      source = at::native::as_strided_tensorimpl(input, input.sizes(), input.strides(), input.storage_offset());
      source._set_conj(false); source._set_neg(false); target = source;
    }
    std::vector<int64_t> bits(2);
    std::memcpy(bits.data(), words, sizeof(words));
    scientific_call(483, {source, expanded_mask, target}, bits, {});
  });
  m.def("check_inplace", [](const at::Tensor& out, const at::Tensor& input) {
    at::assert_no_internal_overlap(out);
    at::assert_no_partial_overlap(out, input);
  });
  m.def("check_index_output", [](const at::Tensor& out, const at::Tensor& source, const at::Tensor& index) {
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, source);
    at::assert_no_overlap(out, index);
  });
  m.def("prepare_index_output", [](at::Tensor out, const std::vector<int64_t>& shape) {
    validate(out.device());
    if (at::native::resize_output_check(out, shape)) {
      auto* impl = out.unsafeGetTensorImpl();
      const auto bytes = at::detail::computeStorageNbytesContiguous(
          shape, out.element_size(), out.storage_offset());
      auto storage = out.storage();
      if (bytes > storage.nbytes()) {
        TORCH_CHECK(storage.resizable(), "index output storage is not resizable");
        storage.set_data_ptr_noswap(allocator.allocate(bytes));
        storage.set_nbytes(bytes);
      }
      impl->set_sizes_contiguous(shape);
    }
  });
  m.def("fill", [](at::Tensor out, pybind11::object value) {
    static torch::PythonArgParser parser({"fill(Scalar value)"});
    torch::ParsedArgs<1> parsed;
    auto args = pybind11::make_tuple(value);
    auto scalar = parser.parse(args.ptr(), nullptr, parsed).scalar(0);
    ::fill(out, scalar);
  });
  m.def("synchronize", synchronize);
}
