#pragma once

#include <cuda.h>
#include <vector>
#include <stdexcept>

#include "../../common/common.cuh"

namespace kittens {
namespace detail {
namespace vmm {

// Intra-node shareable handle type
// This makes the handle shareable with cuMemExportToShareableHandle/cuMemImportFromShareableHandle
static constexpr CUmemAllocationHandleType HANDLE_TYPE = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

typedef CUmemGenericAllocationHandle handle;

__host__ inline static void vm_alloc(
    CUmemGenericAllocationHandle *handle,
    size_t *allocated_size,
    const size_t size,
    const int device_id
) {
    CUmemAllocationProp prop = {};
    prop.location.id = device_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = HANDLE_TYPE;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    size_t granularity;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    *allocated_size = (size + granularity - 1) / granularity * granularity; // round-up

    CUCHECK(cuMemCreate(handle, *allocated_size, &prop, 0));
}

__host__ inline static void vm_map(
    void **ptr,
    const CUmemGenericAllocationHandle &handle,
    const size_t size
) {
    CUdeviceptr device_ptr;
    CUCHECK(cuMemAddressReserve(&device_ptr, size, 0, 0, 0));
    CUCHECK(cuMemMap(device_ptr, size, 0, handle, 0));
    *ptr = (void *)device_ptr;
}

__host__ inline static void vm_set_access(
    void *ptr,
    const size_t size,
    const int num_devices
) {
    std::vector<CUmemAccessDesc> descs(num_devices);
    for (int i = 0; i < num_devices; i++) {
        descs[i].location.id = i;
        descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CUCHECK(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, descs.data(), num_devices));
}

__host__ inline static void vm_retrieve_handle(
    CUmemGenericAllocationHandle *handle,
    void *ptr
) {
    // Every call to this requires a corresponding call to cuMemRelease
    CUCHECK(cuMemRetainAllocationHandle(handle, ptr));
}

__host__ inline static void vm_unmap(
    void *ptr,
    const size_t size
) {
    CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size)); 
    CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
}

__host__ inline static void vm_free(CUmemGenericAllocationHandle &handle) {
    // It is recommended to free the handle ASAP; the backing memory will
    // only be freed when all handles AND address mappings are released
    CUCHECK(cuMemRelease(handle));
}

__host__ inline static void vm_alloc_map_set_access(
    void **ptr,
    size_t *allocated_size,
    const size_t size,
    const int device_id,
    const int num_devices
) {
    CUmemGenericAllocationHandle handle;
    vm_alloc(&handle, allocated_size, size, device_id);
    vm_map(ptr, handle, *allocated_size);
    vm_set_access(*ptr, *allocated_size, num_devices);
    vm_free(handle); // release the handle ASAP
}

__host__ inline static void multicast_check(const int device_id) {
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));

    int multicast_supported;
    CUresult result = cuDeviceGetAttribute(
        &multicast_supported,
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        device
    );

    if (!multicast_supported)
        throw std::runtime_error("Device does not support multicast");
}

__host__ inline static void multicast_create_handle(
    CUmemGenericAllocationHandle *handle,
    size_t *allocated_size,
    const size_t size,
    const int num_devices
) {
    if (num_devices <= 1)
        throw std::runtime_error("Multicast requires at least 2 devices");

    CUmulticastObjectProp prop = {};
    prop.numDevices = num_devices;
    prop.handleTypes = HANDLE_TYPE;

    size_t granularity;
    CUCHECK(cuMulticastGetGranularity(&granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    *allocated_size = (size + granularity - 1) / granularity * granularity;
    prop.size = *allocated_size;

    // After this, the handle must be shared with all processes through MPI, KittensBroker, etc.
    cuMulticastCreate(handle, &prop);
}

__host__ inline static void multicast_bind_device(
    const CUmemGenericAllocationHandle &handle,
    const int device_id
) {
    // All processes must sync after this, before binding any memory
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));
    CUCHECK(cuMulticastAddDevice(handle, device));
}

__host__ inline static void multicast_bind_memory(
    const CUmemGenericAllocationHandle &multicast_handle,
    const CUmemGenericAllocationHandle &memory_handle,
    const size_t size
) {
    // All processes should finish adding device before calling this function
    CUCHECK(cuMulticastBindMem(multicast_handle, 0, memory_handle, 0, size, 0));
}

__host__ inline static void multicast_bind_address(
    const CUmemGenericAllocationHandle &multicast_handle,
    void *ptr,
    const size_t size
) {
    // All processes should finish adding device before calling this function
    CUmemGenericAllocationHandle memory_handle;
    vm_retrieve_handle(&memory_handle, ptr);
    multicast_bind_memory(multicast_handle, memory_handle, size);
    vm_free(memory_handle);
}

__host__ inline static void multicast_unbind_device(
    const CUmemGenericAllocationHandle &handle,
    const size_t size,
    const int device_id
) {
    // Unbinding memory is not needed
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));
    CUCHECK(cuMulticastUnbind(handle, device, 0, size));
}

} // namespace vmm
} // namespace detail
} // namespace kittens
