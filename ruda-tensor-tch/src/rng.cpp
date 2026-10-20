#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef RUDA_TCH_RNG
#include <ATen/ATen.h>
#include <ATen/Context.h>

namespace {
at::Generator generator(int32_t kind, int32_t index) {
    switch (kind) {
    case 0:
        return at::globalContext().defaultGenerator(c10::Device(c10::DeviceType::CPU));
    case 1:
        if (index < 0 || index > std::numeric_limits<c10::DeviceIndex>::max())
            throw std::invalid_argument("Invalid LibTorch CUDA RNG device index");
        return at::globalContext().defaultGenerator(
            c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(index)));
    case 2:
        return at::globalContext().defaultGenerator(c10::Device(c10::DeviceType::MPS));
    default:
        throw std::invalid_argument("LibTorch does not expose an RNG generator for this device");
    }
}
}
#endif

namespace {
thread_local std::string rng_error;
}

extern "C" const char* ruda_tch_rng_error() { return rng_error.c_str(); }

extern "C" uint8_t* ruda_tch_rng_get(int32_t kind, int32_t index, std::size_t* length) {
    rng_error.clear();
    *length = 0;
#ifdef RUDA_TCH_RNG
    try {
        auto gen = generator(kind, index);
        std::lock_guard<std::mutex> lock(gen.mutex());
        auto state = gen.get_state().contiguous();
        if (!state.device().is_cpu() || state.scalar_type() != at::kByte)
            throw std::runtime_error("LibTorch RNG returned a non-byte CPU state");
        const auto size = static_cast<std::size_t>(state.numel());
        auto bytes = std::make_unique<uint8_t[]>(size);
        std::memcpy(bytes.get(), state.const_data_ptr<uint8_t>(), size);
        *length = size;
        return bytes.release();
    } catch (const std::exception& error) { rng_error = error.what(); }
      catch (...) { rng_error = "Unknown LibTorch RNG checkpoint error"; }
#else
    (void)kind; (void)index;
    rng_error = "LibTorch RNG checkpoint requires LibTorch headers at build time; set LIBTORCH or LIBTORCH_USE_PYTORCH";
#endif
    return nullptr;
}

extern "C" void ruda_tch_rng_free(uint8_t* bytes) { delete[] bytes; }

extern "C" int32_t ruda_tch_rng_set(
    int32_t kind, int32_t index, const uint8_t* bytes, std::size_t length) {
    rng_error.clear();
#ifdef RUDA_TCH_RNG
    try {
        if (length == 0 || length > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
            throw std::invalid_argument("Invalid LibTorch RNG checkpoint length");
        auto state = at::empty({static_cast<int64_t>(length)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        std::memcpy(state.data_ptr<uint8_t>(), bytes, length);
        auto gen = generator(kind, index);
        std::lock_guard<std::mutex> lock(gen.mutex());
        auto candidate = gen.clone();
        candidate.set_state(state);
        gen.set_state(state);
        return 0;
    } catch (const std::exception& error) { rng_error = error.what(); }
      catch (...) { rng_error = "Unknown LibTorch RNG restore error"; }
#else
    (void)kind; (void)index; (void)bytes; (void)length;
    rng_error = "LibTorch RNG restoration requires LibTorch headers at build time; set LIBTORCH or LIBTORCH_USE_PYTORCH";
#endif
    return -1;
}
