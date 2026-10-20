#include <cstddef>
#include <cstdint>
#include <string>

namespace {
thread_local std::string collective_error;
}

#ifdef RUDA_TCH_CUDA_PEER
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Prepared {
    at::Tensor tensor;
    c10::cuda::CUDAStream stream;
};
struct Rank {
    at::Tensor output, scratch;
    c10::cuda::CUDAStream stream;
    at::cuda::CUDAEvent ready;
    Rank(const Prepared& input, int64_t scratch_elements) : stream(input.stream) {
        c10::cuda::CUDAStreamGuard guard(stream);
        output = input.tensor.clone();
        scratch = at::empty({scratch_elements}, output.options());
    }
};
struct Work {
    std::vector<std::unique_ptr<Rank>> ranks;
    int64_t elements = 0;
    bool completed = false;
    ~Work() {
        // Also retire outstanding peer reads before freeing buffers on error.
        if (!completed) for (const auto& rank : ranks) {
            try { rank->stream.synchronize(); } catch (...) {}
        }
    }
};
}
#endif

extern "C" const char* ruda_tch_collective_error() { return collective_error.c_str(); }
extern "C" int32_t ruda_tch_peer_available() {
#ifdef RUDA_TCH_CUDA_PEER
    return 1;
#else
    return 0;
#endif
}

extern "C" void* ruda_tch_prepare_collective(const void* tensor) {
    collective_error.clear();
#ifdef RUDA_TCH_CUDA_PEER
    try {
        const auto& input = *static_cast<const at::Tensor*>(tensor);
        if (!input.is_cuda()) throw std::runtime_error("ruCCL peer execution requires CUDA tensors");
        c10::cuda::CUDAGuard guard(input.device());
        at::NoGradGuard no_grad;
        return new Prepared{input.contiguous(), c10::cuda::getCurrentCUDAStream(input.get_device())};
    } catch (const std::exception& error) { collective_error = error.what(); }
#else
    (void)tensor;
    collective_error = "ruCCL LibTorch peer execution requires a CUDA LibTorch build and CUDA headers";
#endif
    return nullptr;
}
extern "C" void ruda_tch_drop_prepared(void* tensor) {
#ifdef RUDA_TCH_CUDA_PEER
    delete static_cast<Prepared*>(tensor);
#else
    (void)tensor;
#endif
}

extern "C" void* ruda_tch_peer_begin(const void* const* prepared, std::size_t count,
                                     std::size_t* elements) {
    collective_error.clear();
#ifdef RUDA_TCH_CUDA_PEER
    try {
        if (count == 0) throw std::runtime_error("empty ruCCL peer device group");
        at::NoGradGuard no_grad;
        auto work = std::make_unique<Work>();
        const auto& first = *static_cast<const Prepared*>(prepared[0]);
        work->elements = first.tensor.numel();
        const int64_t scratch_elements = work->elements / count + (work->elements % count != 0);
        for (std::size_t rank = 0; rank < count; ++rank) {
            const auto& input = *static_cast<const Prepared*>(prepared[rank]);
            if (input.tensor.sizes() != first.tensor.sizes() || input.tensor.scalar_type() != first.tensor.scalar_type())
                throw std::runtime_error("ruCCL peer tensor shape/dtype differs between ranks");
            if (input.tensor.get_device() != input.stream.device_index())
                throw std::runtime_error("ruCCL peer producer stream/device mismatch");
            if (!input.tensor.is_floating_point())
                throw std::runtime_error("ruCCL peer reduction requires floating tensors");
            if (count > 1) {
                const auto& next = *static_cast<const Prepared*>(prepared[(rank + 1) % count]);
                c10::cuda::CUDAGuard guard(next.tensor.device());
                int access = 0;
                C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&access, next.tensor.get_device(), input.tensor.get_device()));
                if (!access) throw std::runtime_error("ruCCL CUDA ring edge lacks peer access; host staging is not used");
                c10::cuda::CUDACachingAllocator::enablePeerAccess(
                    next.tensor.get_device(), input.tensor.get_device());
            }
            work->ranks.push_back(std::make_unique<Rank>(input, scratch_elements));
        }
        *elements = static_cast<std::size_t>(work->elements);
        return work.release();
    } catch (const std::exception& error) { collective_error = error.what(); }
#else
    (void)prepared; (void)count; (void)elements;
    collective_error = "ruCCL LibTorch peer execution requires a CUDA LibTorch build and CUDA headers";
#endif
    return nullptr;
}
extern "C" void ruda_tch_peer_drop(void* work) {
#ifdef RUDA_TCH_CUDA_PEER
    delete static_cast<Work*>(work);
#else
    (void)work;
#endif
}

// ruCCL supplies the ring ranges and phase; this adapter has no collective algorithm.
extern "C" int32_t ruda_tch_peer_exchange(void* opaque, const std::size_t* offsets,
                                          const std::size_t* lengths, std::size_t count,
                                          int32_t reduce) {
    collective_error.clear();
#ifdef RUDA_TCH_CUDA_PEER
    try {
        auto& work = *static_cast<Work*>(opaque);
        if (count != work.ranks.size()) throw std::runtime_error("ruCCL peer range count mismatch");
        at::NoGradGuard no_grad;
        // Record all producers before introducing cross-device dependencies.
        for (const auto& rank : work.ranks) rank->ready.record(rank->stream);
        for (std::size_t source = 0; source < count; ++source) {
            auto& from = *work.ranks[source];
            auto& to = *work.ranks[(source + 1) % count];
            if (offsets[source] > static_cast<std::size_t>(work.elements) ||
                lengths[source] > static_cast<std::size_t>(work.elements) - offsets[source] ||
                lengths[source] > static_cast<std::size_t>(to.scratch.numel()))
                throw std::runtime_error("ruCCL peer range is outside device buffers");
            if (lengths[source] == 0) continue;
            c10::cuda::CUDAStreamGuard guard(to.stream);
            from.ready.block(to.stream);
            const auto input = from.output.view({-1}).narrow(0, offsets[source], lengths[source]);
            C10_CUDA_CHECK(cudaMemcpyPeerAsync(to.scratch.data_ptr(), to.stream.device_index(),
                input.data_ptr(), from.stream.device_index(),
                lengths[source] * input.element_size(), to.stream.stream()));
        }
        // Sends and destination updates use distinct chunks within each ring step.
        for (std::size_t source = 0; source < count; ++source) {
            if (lengths[source] == 0) continue;
            auto& to = *work.ranks[(source + 1) % count];
            c10::cuda::CUDAStreamGuard guard(to.stream);
            auto output = to.output.view({-1}).narrow(0, offsets[source], lengths[source]);
            const auto incoming = to.scratch.narrow(0, 0, lengths[source]);
            if (reduce != 0) output.add_(incoming);
            else output.copy_(incoming);
        }
        return 0;
    } catch (const std::exception& error) { collective_error = error.what(); }
#else
    (void)opaque; (void)offsets; (void)lengths; (void)count; (void)reduce;
    collective_error = "ruCCL LibTorch peer execution requires a CUDA LibTorch build and CUDA headers";
#endif
    return -1;
}
extern "C" int32_t ruda_tch_peer_finish(void* opaque, std::size_t count,
                                        int32_t mean, void** results) {
    collective_error.clear();
    for (std::size_t i = 0; i < count; ++i) results[i] = nullptr;
#ifdef RUDA_TCH_CUDA_PEER
    try {
        auto& work = *static_cast<Work*>(opaque);
        if (count != work.ranks.size()) throw std::runtime_error("ruCCL peer output count mismatch");
        at::NoGradGuard no_grad;
        for (const auto& rank : work.ranks) {
            c10::cuda::CUDAStreamGuard guard(rank->stream);
            if (mean != 0) rank->output.div_(static_cast<double>(count));
        }
        for (const auto& rank : work.ranks) rank->stream.synchronize();
        work.completed = true;
        std::vector<std::unique_ptr<at::Tensor>> owned;
        for (const auto& rank : work.ranks) owned.push_back(std::make_unique<at::Tensor>(rank->output));
        for (std::size_t i = 0; i < count; ++i) results[i] = owned[i].release();
        return 0;
    } catch (const std::exception& error) { collective_error = error.what(); }
#else
    (void)opaque; (void)mean;
    collective_error = "ruCCL LibTorch peer execution requires a CUDA LibTorch build and CUDA headers";
#endif
    return -1;
}
