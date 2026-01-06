#pragma once

#include <iostream>
#include <map>
#include <vector>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include "../types/device/vmm.cuh"
#include "../types/device/ipc.cuh"
#include "broker.cuh"

namespace kittens {
namespace py {

/**
 * @brief Distributed tensor wrapper for multi-GPU IPC sharing and multicast.
 *        Can be later used for easy PGL creation right before a kernel call.
 *        Meant to be used as a single object per thread/process.
 */
struct TKParallelTensor {
    inline static std::map<std::pair<int, int>, KittensBroker> brokers_; // lazily initialized

    at::Tensor data_; // for direct access from PyTorch
    std::vector<int64_t> shape_;
    at::ScalarType dtype_;

    std::vector<void *> raw_ptrs_;
    size_t allocated_size_;

    int local_rank_; // identical to device index
    int local_world_size_;

    bool multicast_;
    void *multicast_ptr_;
    size_t multicast_allocated_size_;

    detail::ipc::flavor ipc_flavor_;

    __host__ inline TKParallelTensor(
        const at::Tensor &tensor,
        int local_rank,
        int local_world_size,
        bool multicast
    ) : data_(tensor),
        shape_(tensor.sizes().vec()),
        dtype_(tensor.scalar_type()),
        raw_ptrs_(local_world_size, nullptr),
        allocated_size_(tensor.nbytes()),
        local_rank_(local_rank),
        local_world_size_(local_world_size),
        multicast_(multicast),
        multicast_ptr_(nullptr),
        multicast_allocated_size_(0),
        ipc_flavor_(detail::ipc::flavor::LEGACY) {

        TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
        TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
        TORCH_CHECK(tensor.dim() <= 4, "Only tensors with dim <= 4 are supported for TKParallelTensor");
        TORCH_CHECK(tensor.device().index() == local_rank_, "Tensor device index must match local_rank");
        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");
        TORCH_CHECK(!multicast, "Multicast is not supported for pre-allocated tensors");

        brokers_.try_emplace(
            {local_rank_, local_world_size_},
            local_rank_, local_world_size_
        );

        if (brokers_.size() > 1)
            std::cerr << "WARNING: 2 KittensBroker instances created in the same process. This is not safe." << std::endl;

        c10::cuda::CUDAGuard device_guard(local_rank_);
        exchange_ipc_handles<detail::ipc::flavor::LEGACY>();
    }

    __host__ inline TKParallelTensor(
        const std::vector<int64_t> &shape,
        const at::ScalarType dtype,
        int local_rank,
        int local_world_size,
        bool multicast
    ) : shape_(shape),
        dtype_(dtype),
        raw_ptrs_(local_world_size, nullptr),
        allocated_size_(0),
        local_rank_(local_rank),
        local_world_size_(local_world_size),
        multicast_(multicast),
        multicast_ptr_(nullptr),
        multicast_allocated_size_(0),
        ipc_flavor_(detail::ipc::flavor::VMM) {

        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");

        brokers_.try_emplace(
            {local_rank_, local_world_size_},
            local_rank_, local_world_size_
        );

        if (brokers_.size() > 1)
            std::cerr << "WARNING: 2 KittensBroker instances created in the same process. This is not safe." << std::endl;

        c10::cuda::CUDAGuard device_guard(local_rank_);
        create_shareable_cuda_tensor();
        exchange_ipc_handles<detail::ipc::flavor::VMM>();

        if (multicast_)
            initialize_multicast();
    }

    TKParallelTensor(const TKParallelTensor&) = delete;
    TKParallelTensor& operator=(const TKParallelTensor&) = delete;
    TKParallelTensor& operator=(TKParallelTensor&& other) = delete;

    __host__ inline TKParallelTensor(TKParallelTensor&& other) :
        data_(std::move(other.data_)),
        shape_(std::move(other.shape_)),
        dtype_(std::move(other.dtype_)),
        raw_ptrs_(std::move(other.raw_ptrs_)),
        allocated_size_(other.allocated_size_),
        local_rank_(other.local_rank_),
        local_world_size_(other.local_world_size_),
        multicast_(other.multicast_),
        multicast_ptr_(other.multicast_ptr_),
        multicast_allocated_size_(other.multicast_allocated_size_),
        ipc_flavor_(other.ipc_flavor_) {
        other.data_ = at::Tensor();
        other.shape_.clear();
        other.dtype_ = at::ScalarType::Undefined;
        other.raw_ptrs_.clear();
        other.allocated_size_ = 0;
        other.local_rank_ = -1;
        other.local_world_size_ = -1;
        other.multicast_ = false;
        other.multicast_ptr_ = nullptr;
        other.multicast_allocated_size_ = 0;
    }

    __host__ inline ~TKParallelTensor() {
        destroy();
    }

    __host__ inline at::Tensor data() const {
        return data_;
    }

    __host__ inline void create_shareable_cuda_tensor() {
        c10::cuda::CUDAGuard device_guard(local_rank_);

        TORCH_CHECK(!shape_.empty(), "Shape must be non-empty");
        TORCH_CHECK(shape_.size() <= 4, "Shape must have at most 4 dimensions for TKParallelTensor");
        size_t size = c10::elementSize(dtype_);
        for (auto dim : shape_) {
            TORCH_CHECK(dim > 0, "Size dimensions must be positive");
            size *= static_cast<size_t>(dim);
        }

        void *raw_ptr;
        detail::vmm::vm_alloc_map_set_access(
            &raw_ptr, &allocated_size_, size, local_rank_, local_world_size_);

        // Create local copies for capture
        int local_rank = local_rank_;
        size_t allocated_size = allocated_size_;

        auto deleter = [local_rank, raw_ptr, allocated_size](void* p) mutable {
            if (!p) return;
            c10::cuda::CUDAGuard device_guard(local_rank);
            auto stream = c10::cuda::getCurrentCUDAStream().stream();
            CUDACHECK(cudaStreamSynchronize(stream));
            detail::vmm::vm_unmap(raw_ptr, allocated_size);
        };

        at::TensorOptions options = at::TensorOptions()
            .dtype(dtype_)
            .device(at::kCUDA, local_rank_);

        data_ = at::from_blob(raw_ptr, shape_, std::move(deleter), options);
    }

    template <detail::ipc::flavor IPC_FLAVOR>
    __host__ inline void exchange_ipc_handles() {
        using handle_t = detail::ipc::handle<IPC_FLAVOR>;

        // Get IPC handle
        detail::ipc::check_support(local_rank_);
        void *raw_ptr = reinterpret_cast<void *>(data_.data_ptr());
        handle_t ipc_handle;
        detail::ipc::export_handle(&ipc_handle, raw_ptr);

        // Exchange IPC handles
        std::vector<handle_t> all_ipc_handles(local_world_size_);
        if constexpr (IPC_FLAVOR == detail::ipc::flavor::LEGACY) {
            brokers_.at({local_rank_, local_world_size_}).exchange_data(
                reinterpret_cast<void *>(all_ipc_handles.data()),
                reinterpret_cast<void *>(&ipc_handle),
                sizeof(handle_t)
            );
        } else if constexpr (IPC_FLAVOR == detail::ipc::flavor::VMM) {
            brokers_.at({local_rank_, local_world_size_}).exchange_fds(
                reinterpret_cast<int *>(all_ipc_handles.data()),
                ipc_handle.handle_
            );
        } else {
            throw std::runtime_error("Invalid IPC flavor");
        }

        // Import IPC handles
        for (int i = 0; i < local_world_size_; i++) {
            if (i == local_rank_)
                raw_ptrs_[i] = raw_ptr;
            else
                detail::ipc::import_handle(&raw_ptrs_[i], all_ipc_handles[i], allocated_size_, local_world_size_);
        }
    }

    __host__ inline void initialize_multicast() {
        using handle_t = detail::ipc::handle<detail::ipc::flavor::VMM>;

        detail::vmm::multicast_check(local_rank_);
        detail::ipc::check_support(local_rank_);
        detail::vmm::handle multicast_handle;

        if (local_rank_ == 0) {
            // Create multicast handle; only a single rank should create MC handle
            detail::vmm::multicast_create_handle(
                &multicast_handle,
                &multicast_allocated_size_,
                allocated_size_,
                local_world_size_
            );

            // Currently, non-rank-0 path assumes allocated_size_ == multicast_allocated_size_
            if (allocated_size_ != multicast_allocated_size_)
                throw std::runtime_error("Multicast allocated size does not match memory allocated size");

            // Get IPC handle
            handle_t ipc_handle;
            detail::ipc::export_handle(&ipc_handle, multicast_handle);

            // Broadcast the IPC multicast handle
            brokers_.at({local_rank_, local_world_size_}).broadcast_fd(nullptr, ipc_handle.handle_, 0);
        } else {
            // Receive the IPC multicast handle from rank 0
            handle_t ipc_handle;
            brokers_.at({local_rank_, local_world_size_}).broadcast_fd(&ipc_handle.handle_, -1, 0);
            multicast_allocated_size_ = allocated_size_;
            detail::ipc::import_handle(&multicast_handle, ipc_handle, multicast_allocated_size_, local_world_size_);
        }

        // Add all devices to the MC handle. Must sync
        detail::vmm::multicast_bind_device(multicast_handle, local_rank_);
        brokers_.at({local_rank_, local_world_size_}).sync(); // must ensure all devices are added

        // Bind all memory to the MC handle and map to a virtual address; must be done after adding all devices
        detail::vmm::handle memory_handle;
        detail::vmm::vm_retrieve_handle(&memory_handle, raw_ptrs_[local_rank_]);
        detail::vmm::multicast_bind_memory(multicast_handle, memory_handle, allocated_size_);
        brokers_.at({local_rank_, local_world_size_}).sync();

        // Map virtual address to multicast handle and set access; must be done after adding all devices
        detail::vmm::vm_map(&multicast_ptr_, multicast_handle, multicast_allocated_size_);
        detail::vmm::vm_set_access(multicast_ptr_, multicast_allocated_size_, local_world_size_);

        // Free the handles immediately
        detail::vmm::vm_free(multicast_handle);
        detail::vmm::vm_free(memory_handle);
    }

    __host__ inline void destroy() {
        // 1. Multicast cleanup
        if (multicast_ && multicast_ptr_) {
            brokers_.at({local_rank_, local_world_size_}).sync();
            detail::vmm::handle multicast_handle;
            detail::vmm::vm_retrieve_handle(&multicast_handle, multicast_ptr_);
            detail::vmm::vm_unmap(multicast_ptr_, multicast_allocated_size_);
            detail::vmm::multicast_unbind_device(multicast_handle, multicast_allocated_size_, local_rank_);
            brokers_.at({local_rank_, local_world_size_}).sync();
            detail::vmm::vm_free(multicast_handle);
        }

        // 2. Imported handle cleanup
        for (int i = 0; i < local_world_size_; i++) {
            if (i != local_rank_ && i < raw_ptrs_.size()) {
                if (ipc_flavor_ == detail::ipc::flavor::LEGACY) {
                    detail::ipc::free_handle<detail::ipc::flavor::LEGACY>(raw_ptrs_[i], allocated_size_);
                } else if (ipc_flavor_ == detail::ipc::flavor::VMM) {
                    detail::ipc::free_handle<detail::ipc::flavor::VMM>(raw_ptrs_[i], allocated_size_);
                } else {
                    throw std::runtime_error("Invalid IPC flavor");
                }
            }
        }
        brokers_.at({local_rank_, local_world_size_}).sync(); // must sync before destroying the tensor

        // 3. Tensor cleanup
        if (data_.defined())
            data_.reset(); // properly decreases the ref count

        // 4. Member variables cleanup
        shape_.clear();
        dtype_ = at::ScalarType::Undefined;
        raw_ptrs_.clear();
        allocated_size_ = 0;
        local_rank_ = -1;
        local_world_size_ = -1;
        multicast_ = false;
        multicast_ptr_ = nullptr;
        multicast_allocated_size_ = 0;
    }
};

} // namespace py
} // namespace kittens

#define BIND_TK_PARALLEL_TENSOR(m) \
    pybind11::class_<kittens::py::TKParallelTensor>(m, "TKParallelTensor") \
        .def(pybind11::init<const at::Tensor&, int, int, bool>(), \
             pybind11::arg("tensor"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
        .def(pybind11::init<const std::vector<int64_t>&, const at::ScalarType&, int, int, bool>(), \
             pybind11::arg("shape"), \
             pybind11::arg("dtype"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
        .def("data", &kittens::py::TKParallelTensor::data) \
        .def_readonly("data_", &kittens::py::TKParallelTensor::data_) \
        .def_readonly("local_rank_", &kittens::py::TKParallelTensor::local_rank_) \
        .def_readonly("local_world_size_", &kittens::py::TKParallelTensor::local_world_size_)
