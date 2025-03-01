#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/csrc/PyInterpreter.h>
#include <ATen/OpaqueTensorImpl.h>

// register guard
namespace at {
namespace detail {
//C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
// NOTE: pytorch's no-op class throws error on backwards with events/streams
// TODO: why are there events in autograd?
struct CustomNoOpDeviceGuardImpl : public c10::impl::DeviceGuardImplInterface
{
  static const DeviceType D = DeviceType::PrivateUse1;
  CustomNoOpDeviceGuardImpl() = default;
  DeviceType type() const override {
    return D;
  }
  Device exchangeDevice(Device) const override {
    return Device(D, 0); // no-op
  }
  Device getDevice() const override {
    return Device(D, 0);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device) const noexcept override {
    // no-op
  }
  Stream getStream(Device) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getDefaultStream(Device) const override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const override {
    // no-op
    (void)isHighPriority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getNewStream(Device, int priority = 0) const override {
    // no-op
    (void)priority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
  // Event-related functions
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.");
  }
  void block(void* /*event*/, const Stream& /*stream*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
    return true;
  }
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}
  // Stream-related functions
  bool queryStream(const Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomNoOpDeviceGuardImpl);
}

template <typename OpaqueHandle>
struct TinyOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  TinyOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides)
      : OpaqueTensorImpl<OpaqueHandle>(key_set, data_type, device, opaque_handle, sizes)
        { this->sizes_and_strides_.set_strides(strides); }
};
}

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  // NOTE: no idea what this is
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }
};

int register_hook() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());
  return 0;
}
int temp_register_hook = register_hook();

at::Tensor wrap_tensor(py::object &py_obj, c10::ScalarType dtype) {
  // TODO: we have to get the dtype and the shape from the tinygrad Tensor
  std::vector<int64_t> sizes = py_obj.attr("shape").cast<std::vector<int64_t>>();

  // Last dimension stride is 1 for contiguous row-major layout
  std::vector<int64_t> strides(sizes.size());
  if (sizes.size() >= 1) {
    strides[sizes.size() - 1] = 1;

    // Compute strides from right to left
    for (int64_t i = sizes.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  return at::detail::make_tensor<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    c10::scalarTypeToTypeMeta(dtype),
    at::Device(at::kPrivateUse1),
    std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
    sizes, strides);
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  auto* opaque_impl = static_cast<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>*>(impl);
  std::shared_ptr<c10::SafePyObject> tiny = opaque_impl->opaque_handle();
  return py::reinterpret_borrow<py::object>(tiny->ptr(getPyInterpreter()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
}
