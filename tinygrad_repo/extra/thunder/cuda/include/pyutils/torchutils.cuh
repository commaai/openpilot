#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>

#include "kittens.cuh"
#include "parallel_tensor.cuh"

namespace kittens {
namespace py {

template <typename Config>
concept has_min_blocks_per_sm = requires { std::integral_constant<int, int(Config::MIN_BLOCKS_PER_SM)>{}; };

template <typename Config>
consteval int min_blocks_per_sm() {
    if constexpr(has_min_blocks_per_sm<Config>)
        return Config::MIN_BLOCKS_PER_SM;
    else
        return 1;
}

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
void global_kernel_unclustered(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
__cluster_dims__(Config::CLUSTER_SIZE)
void global_kernel_clustered(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Layout>
static inline void tensor_check(const at::Tensor &t) {
    TORCH_CHECK(t.is_cuda(), "Tensor must be on CUDA device")
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous")
    TORCH_CHECK(t.dim() <= 4, "Expected Tensor.dim() <= 4");

    if constexpr (std::is_same_v<typename Layout::dtype, char>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Char, "Tensor has invalid dtype (expected int8)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, short>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Short, "Tensor has invalid dtype (expected int16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, int>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Int, "Tensor has invalid dtype (expected int32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, long>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Long, "Tensor has invalid dtype (expected int64)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e4m3>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, "Tensor has invalid dtype (expected fp8e4m3)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e5m2>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e5m2, "Tensor has invalid dtype (expected fp8e5m2)");
#ifdef KITTENS_BLACKWELL
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e8m0>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Byte, "Tensor has invalid dtype (expected fp8e8m0 represented as uint8)");
#endif
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::bf16>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::BFloat16, "Tensor has invalid dtype (expected bfloat16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::half>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Half, "Tensor has invalid dtype (expected float16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, float>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float, "Tensor has invalid dtype (expected float32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, double>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Double, "Tensor has invalid dtype (expected float64)");
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

template <kittens::ducks::pgl::all PGL>
static inline void parallel_tensor_check(const TKParallelTensor& t) {
    tensor_check<PGL>(t.data_);
    TORCH_CHECK(t.data_.sizes().vec() == t.shape_, "Shape mismatch between TKParallelTensor and the underlying tensor");
    TORCH_CHECK(t.data_.dtype() == t.dtype_, "Dtype mismatch between TKParallelTensor and the underlying tensor");
    TORCH_CHECK(t.raw_ptrs_.size() == PGL::num_devices, "Number of devices mismatch between PGL and TKParallelTensor");
    TORCH_CHECK(t.local_rank_ == t.data_.device().index(), "Current tensor device index mismatch within TKParallelTensor");
    TORCH_CHECK(t.local_world_size_ == PGL::num_devices, "Number of devices mismatch between PGL and TKParallelTensor");
    TORCH_CHECK(t.multicast_ == PGL::multicast, "Multicast mismatch between PGL and TKParallelTensor");
    TORCH_CHECK(t.raw_ptrs_[t.local_rank_] == reinterpret_cast<void *>(t.data_.data_ptr()), "Current tensor data pointer not found in TKParallelTensor's raw_ptrs_");
}

template <kittens::ducks::gl::all GL>
static inline GL tensor_to_gl(const at::Tensor &t) {
    tensor_check<GL>(t);

    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.dim()); ++i)
        shape[4 - t.dim() + i] = static_cast<int>(t.size(i));

    uint64_t data_ptr = reinterpret_cast<uint64_t>(t.data_ptr());

    return ::kittens::make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
}

template <kittens::ducks::pgl::all PGL>
static inline PGL parallel_tensor_to_pgl(TKParallelTensor &t) {
    parallel_tensor_check<PGL>(t);

    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.data_.dim()); ++i) {
        shape[4 - t.data_.dim() + i] = static_cast<int>(t.data_.size(i));
    }

    if constexpr (PGL::multicast)
        return ::kittens::make_pgl<PGL>(
            reinterpret_cast<uint64_t>(t.multicast_ptr_), reinterpret_cast<uint64_t *>(t.raw_ptrs_.data()), shape[0], shape[1], shape[2], shape[3]);
    else
        return ::kittens::make_pgl<PGL>(
            reinterpret_cast<uint64_t *>(t.raw_ptrs_.data()), shape[0], shape[1], shape[2], shape[3]);
}

template <kittens::ducks::gl::all GL>
static inline GL make_fake_gl(const int batch, const int depth, const int rows, const int cols) {
    return ::kittens::make_gl<GL>(reinterpret_cast<uint64_t>(nullptr), batch, depth, rows, cols);
}

static inline void _device_check(const at::Tensor& first, const at::Tensor& second) {
    TORCH_CHECK(first.device() == second.device(), "All tensors must be on the same device");
}

template <typename T1, typename... Ts>
static inline void device_check(const T1& first, const Ts&... rest) {
    (_device_check(first, rest), ...);
}

static inline void _parallel_tensor_check(const TKParallelTensor& first, const TKParallelTensor& second) {
    TORCH_CHECK(first.local_rank_ == second.local_rank_, "All parallel tensors must have the same local_rank");
    TORCH_CHECK(first.local_world_size_ == second.local_world_size_, "All parallel tensors must have the same local_world_size");
}

template <typename T1, typename... Ts>
static inline void parallel_tensor_check(const T1& first, const Ts&... rest) {
    (_parallel_tensor_check(first, rest), ...);
}

template <typename Config>
concept static_grid = requires { Config::NUM_BLOCKS; };

template <typename Config>
concept static_block = requires { Config::NUM_THREADS; };

template <typename Config>
concept static_dynamic_shared_memory = requires { Config::DYNAMIC_SHARED_MEMORY; };

template <typename Config, typename Globals, auto Kernel>
static inline void launch_kernel(const Globals &G) {
    dim3 grid;
    if constexpr (static_grid<Config>)
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();

    dim3 block;
    if constexpr (static_block<Config>)
        block = dim3{Config::NUM_THREADS, 1, 1};
    else
        block = G.block();

    int dynamic_shared_memory;
    if constexpr (static_dynamic_shared_memory<Config>)
        dynamic_shared_memory = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    else
        dynamic_shared_memory = G.dynamic_shared_memory();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if constexpr (Config::CLUSTER_SIZE <= 1) {
        CUDACHECK(cudaFuncSetAttribute(global_kernel_unclustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        global_kernel_unclustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    } else {
        CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        global_kernel_clustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    }
}

} // namespace py
} // namespace kittens
