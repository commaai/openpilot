// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H

namespace Eigen {

static const int kCudaScratchSize = 1024;

// This defines an interface that GPUDevice can take to use
// CUDA streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() {}

  virtual const cudaStream_t& stream() const = 0;
  virtual const cudaDeviceProp& deviceProperties() const = 0;

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;

  // Return a scratchpad buffer of size 1k
  virtual void* scratchpad() const = 0;

  // Return a semaphore. The semaphore is initially initialized to 0, and
  // each kernel using it is responsible for resetting to 0 upon completion
  // to maintain the invariant that the semaphore is always equal to 0 upon
  // each kernel start.
  virtual unsigned int* semaphore() const = 0;
};

static cudaDeviceProp* m_deviceProperties;
static bool m_devicePropInitialized = false;

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    // Attempts to ensure proper behavior in the case of multiple threads
    // calling this function simultaneously. This would be trivial to
    // implement if we could use std::mutex, but unfortunately mutex don't
    // compile with nvcc, so we resort to atomics and thread fences instead.
    // Note that if the caller uses a compiler that doesn't support c++11 we
    // can't ensure that the initialization is thread safe.
#if __cplusplus >= 201103L
    static std::atomic<bool> first(true);
    if (first.exchange(false)) {
#else
    static bool first = true;
    if (first) {
      first = false;
#endif
      // We're the first thread to reach this point.
      int num_devices;
      cudaError_t status = cudaGetDeviceCount(&num_devices);
      if (status != cudaSuccess) {
        std::cerr << "Failed to get the number of CUDA devices: "
                  << cudaGetErrorString(status)
                  << std::endl;
        assert(status == cudaSuccess);
      }
      m_deviceProperties = new cudaDeviceProp[num_devices];
      for (int i = 0; i < num_devices; ++i) {
        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
        if (status != cudaSuccess) {
          std::cerr << "Failed to initialize CUDA device #"
                    << i
                    << ": "
                    << cudaGetErrorString(status)
                    << std::endl;
          assert(status == cudaSuccess);
        }
      }

#if __cplusplus >= 201103L
      std::atomic_thread_fence(std::memory_order_release);
#endif
      m_devicePropInitialized = true;
    } else {
      // Wait for the other thread to inititialize the properties.
      while (!m_devicePropInitialized) {
#if __cplusplus >= 201103L
        std::atomic_thread_fence(std::memory_order_acquire);
#endif
        sleep(1);
      }
    }
  }
}

static const cudaStream_t default_stream = cudaStreamDefault;

class CudaStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
    cudaGetDevice(&device_);
    initializeDeviceProp();
  }
  // Use the default stream on the specified device
  CudaStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
    initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
      : stream_(stream), device_(device), scratch_(NULL), semaphore_(NULL) {
    if (device < 0) {
      cudaGetDevice(&device_);
    } else {
      int num_devices;
      cudaError_t err = cudaGetDeviceCount(&num_devices);
      EIGEN_UNUSED_VARIABLE(err)
      assert(err == cudaSuccess);
      assert(device < num_devices);
      device_ = device;
    }
    initializeDeviceProp();
  }

  virtual ~CudaStreamDevice() {
    if (scratch_) {
      deallocate(scratch_);
    }
  }

  const cudaStream_t& stream() const { return *stream_; }
  const cudaDeviceProp& deviceProperties() const {
    return m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    cudaError_t err = cudaSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
    void* result;
    err = cudaMalloc(&result, num_bytes);
    assert(err == cudaSuccess);
    assert(result != NULL);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    cudaError_t err = cudaSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
    assert(buffer != NULL);
    err = cudaFree(buffer);
    assert(err == cudaSuccess);
  }

  virtual void* scratchpad() const {
    if (scratch_ == NULL) {
      scratch_ = allocate(kCudaScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  virtual unsigned int* semaphore() const {
    if (semaphore_ == NULL) {
      char* scratch = static_cast<char*>(scratchpad()) + kCudaScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      cudaError_t err = cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
      EIGEN_UNUSED_VARIABLE(err)
      assert(err == cudaSuccess);
    }
    return semaphore_;
  }

 private:
  const cudaStream_t* stream_;
  int device_;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

struct GpuDevice {
  // The StreamInterface is not owned: the caller is
  // responsible for its initialization and eventual destruction.
  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
    eigen_assert(stream);
  }
  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
    eigen_assert(stream);
  }
  // TODO(bsteiner): This is an internal API, we should not expose it.
  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
    return stream_->stream();
  }

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return stream_->allocate(num_bytes);
  }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    stream_->deallocate(buffer);
  }

  EIGEN_STRONG_INLINE void* scratchpad() const {
    return stream_->scratchpad();
  }

  EIGEN_STRONG_INLINE unsigned int* semaphore() const {
    return stream_->semaphore();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
                                      stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
  eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
  }

  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
  eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME
    return 48*1024;
  }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on cuda devices.
    return firstLevelCacheSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    cudaError_t err = cudaStreamSynchronize(stream_->stream());
    if (err != cudaSuccess) {
      std::cerr << "Error detected in CUDA stream: "
                << cudaGetErrorString(err)
                << std::endl;
      assert(err == cudaSuccess);
    }
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
    return stream_->deviceProperties().multiProcessorCount;
  }
  EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
    return stream_->deviceProperties().maxThreadsPerBlock;
  }
  EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
    return stream_->deviceProperties().maxThreadsPerMultiProcessor;
  }
  EIGEN_STRONG_INLINE int sharedMemPerBlock() const {
    return stream_->deviceProperties().sharedMemPerBlock;
  }
  EIGEN_STRONG_INLINE int majorDeviceVersion() const {
    return stream_->deviceProperties().major;
  }
  EIGEN_STRONG_INLINE int minorDeviceVersion() const {
    return stream_->deviceProperties().minor;
  }

  EIGEN_STRONG_INLINE int maxBlocks() const {
    return max_blocks_;
  }

  // This function checks if the CUDA runtime recorded an error for the
  // underlying stream device.
  inline bool ok() const {
#ifdef __CUDACC__
    cudaError_t error = cudaStreamQuery(stream_->stream());
    return (error == cudaSuccess) || (error == cudaErrorNotReady);
#else
    return false;
#endif
  }

 private:
  const StreamInterface* stream_;
  int max_blocks_;
};

#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
  (kernel) <<< (gridsize), (blocksize), (sharedmem), (device).stream() >>> (__VA_ARGS__);   \
  assert(cudaGetLastError() == cudaSuccess);


// FIXME: Should be device and kernel specific.
#ifdef __CUDACC__
static EIGEN_DEVICE_FUNC inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
#ifndef __CUDA_ARCH__
  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
  EIGEN_UNUSED_VARIABLE(status)
  assert(status == cudaSuccess);
#else
  EIGEN_UNUSED_VARIABLE(config)
#endif
}
#endif

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
