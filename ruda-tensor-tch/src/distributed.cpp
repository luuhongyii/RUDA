#include <cstddef>
#include <cstdint>
#include <string>

namespace {
thread_local std::string collective_error;
}

#ifdef RUDA_TCH_NCCL
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/cuda/nccl.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Prepared {
    at::Tensor tensor;
    c10::cuda::CUDAStream stream;
};
}
#endif

extern "C" const char* ruda_tch_collective_error() { return collective_error.c_str(); }

extern "C" int32_t ruda_tch_nccl_available() {
#ifdef RUDA_TCH_NCCL
    return 1;
#else
    return 0;
#endif
}

extern "C" void* ruda_tch_prepare_collective(const void* tensor) {
    collective_error.clear();
#ifdef RUDA_TCH_NCCL
    try {
        const auto& input = *static_cast<const at::Tensor*>(tensor);
        if (!input.is_cuda()) throw std::runtime_error("native collective requires CUDA tensors");
        c10::cuda::CUDAGuard guard(input.device());
        at::NoGradGuard no_grad;
        return new Prepared{input.contiguous(), c10::cuda::getCurrentCUDAStream(input.get_device())};
    } catch (const std::exception& error) { collective_error = error.what(); }
#else
    (void)tensor;
    collective_error = "LibTorch CUDA device adapter requires a Linux NCCL-enabled LibTorch build";
#endif
    return nullptr;
}

extern "C" void ruda_tch_drop_prepared(void* tensor) {
#ifdef RUDA_TCH_NCCL
    delete static_cast<Prepared*>(tensor);
#else
    (void)tensor;
#endif
}

extern "C" int32_t ruda_tch_all_reduce(const void* const* prepared, std::size_t count,
                                      int32_t mean, void** results) {
    collective_error.clear();
    for (std::size_t i = 0; i < count; ++i) results[i] = nullptr;
#ifdef RUDA_TCH_NCCL
    try {
        if (count == 0) throw std::runtime_error("empty native device collective");
        at::NoGradGuard no_grad;
        std::vector<at::Tensor> inputs;
        std::vector<at::Tensor> outputs;
        torch::cuda::nccl::stream_list streams;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& value = *static_cast<const Prepared*>(prepared[i]);
            c10::cuda::CUDAStreamGuard guard(value.stream);
            inputs.push_back(value.tensor);
            outputs.push_back(at::empty_like(value.tensor));
            streams.push_back(value.stream);
        }
        for (const auto& value : inputs) {
            if (value.sizes() != inputs[0].sizes() || value.scalar_type() != inputs[0].scalar_type())
                throw std::runtime_error("collective tensor shape/dtype differs between ranks");
        }
        torch::cuda::nccl::all_reduce(inputs, outputs,
            static_cast<int32_t>(torch::cuda::nccl::ncclRedOp::Sum), streams);
        for (std::size_t i = 0; i < count; ++i) {
            c10::cuda::CUDAStreamGuard guard(*streams[i]);
            if (mean != 0) outputs[i].div_(static_cast<double>(count));
            streams[i]->synchronize();
        }
        std::vector<std::unique_ptr<at::Tensor>> owned;
        for (auto& output : outputs) owned.push_back(std::make_unique<at::Tensor>(output));
        for (std::size_t i = 0; i < count; ++i) results[i] = owned[i].release();
        return 0;
    } catch (const std::exception& error) { collective_error = error.what(); }
#else
    (void)prepared; (void)mean;
    collective_error = "LibTorch CUDA device adapter requires a Linux NCCL-enabled LibTorch build";
#endif
    return -1;
}
