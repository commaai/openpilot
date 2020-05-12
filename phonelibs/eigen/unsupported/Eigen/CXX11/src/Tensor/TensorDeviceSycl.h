// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>

//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_SYCL) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H

namespace Eigen {
struct SyclDevice {
  /// class members
  /// sycl queue
  mutable cl::sycl::queue m_queue;
  /// std::map is the container used to make sure that we create only one buffer
  /// per pointer. The lifespan of the buffer now depends on the lifespan of SyclDevice.
  /// If a non-read-only pointer is needed to be accessed on the host we should manually deallocate it.
  mutable std::map<const void *, std::shared_ptr<void>> buffer_map;
  /// creating device by using selector
  template<typename dev_Selector> SyclDevice(dev_Selector s)
  :
#ifdef EIGEN_EXCEPTIONS
  m_queue(cl::sycl::queue(s, [=](cl::sycl::exception_list l) {
    for (const auto& e : l) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception e) {
          std::cout << e.what() << std::endl;
        }
    }
  }))
#else
  m_queue(cl::sycl::queue(s))
#endif
  {}
  // destructor
  ~SyclDevice() { deallocate_all(); }

  template <typename T> void deallocate(T *p) const {
    auto it = buffer_map.find(p);
    if (it != buffer_map.end()) {
      buffer_map.erase(it);
      internal::aligned_free(p);
    }
  }
  void deallocate_all() const {
    std::map<const void *, std::shared_ptr<void>>::iterator it=buffer_map.begin();
    while (it!=buffer_map.end()) {
      auto p=it->first;
      buffer_map.erase(it);
      internal::aligned_free(const_cast<void*>(p));
      it=buffer_map.begin();
    }
    buffer_map.clear();
  }

  /// creation of sycl accessor for a buffer. This function first tries to find
  /// the buffer in the buffer_map. If found it gets the accessor from it, if not,
  ///the function then adds an entry by creating a sycl buffer for that particular pointer.
  template <cl::sycl::access::mode AcMd, typename T> inline cl::sycl::accessor<T, 1, AcMd, cl::sycl::access::target::global_buffer>
  get_sycl_accessor(size_t num_bytes, cl::sycl::handler &cgh, const T * ptr) const {
    return (get_sycl_buffer<T>(num_bytes, ptr)->template get_access<AcMd, cl::sycl::access::target::global_buffer>(cgh));
  }

  template<typename T> inline  std::pair<std::map<const void *, std::shared_ptr<void>>::iterator,bool> add_sycl_buffer(const T *ptr, size_t num_bytes) const {
    using Type = cl::sycl::buffer<T, 1>;
    std::pair<std::map<const void *, std::shared_ptr<void>>::iterator,bool> ret = buffer_map.insert(std::pair<const void *, std::shared_ptr<void>>(ptr, std::shared_ptr<void>(new Type(cl::sycl::range<1>(num_bytes)),
      [](void *dataMem) { delete static_cast<Type*>(dataMem); })));
    (static_cast<Type*>(buffer_map.at(ptr).get()))->set_final_data(nullptr);
    return ret;
  }

  template <typename T> inline cl::sycl::buffer<T, 1>* get_sycl_buffer(size_t num_bytes,const T * ptr) const {
    return static_cast<cl::sycl::buffer<T, 1>*>(add_sycl_buffer(ptr, num_bytes).first->second.get());
  }

  /// allocating memory on the cpu
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void *allocate(size_t) const {
    return internal::aligned_malloc(8);
  }

  // some runtime conditions that can be applied here
  bool isDeviceSuitable() const { return true; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void *dst, const void *src, size_t n) const {
    ::memcpy(dst, src, n);
  }

  template<typename T> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(T *dst, const T *src, size_t n) const {
    auto host_acc= (static_cast<cl::sycl::buffer<T, 1>*>(add_sycl_buffer(dst, n).first->second.get()))-> template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>();
    memcpy(host_acc.get_pointer(), src, n);
  }
 /// whith the current implementation of sycl, the data is copied twice from device to host. This will be fixed soon.
  template<typename T> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(T *dst, const T *src, size_t n) const {
    auto it = buffer_map.find(src);
    if (it != buffer_map.end()) {
      auto host_acc= (static_cast<cl::sycl::buffer<T, 1>*>(it->second.get()))-> template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
      memcpy(dst,host_acc.get_pointer(),  n);
    } else{
      eigen_assert("no device memory found. The memory might be destroyed before creation");
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void *buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
  return 1;
  }
};

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
