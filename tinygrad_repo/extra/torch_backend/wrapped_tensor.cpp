#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/csrc/PyInterpreter.h>
#include <ATen/OpaqueTensorImpl.h>

// register guard
namespace at {
namespace detail {
C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
}
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

  return at::detail::make_tensor<at::OpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    c10::scalarTypeToTypeMeta(dtype),
    at::Device(at::kPrivateUse1),
    std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
    sizes);
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  auto* opaque_impl = static_cast<at::OpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>*>(impl);
  std::shared_ptr<c10::SafePyObject> tiny = opaque_impl->opaque_handle();
  return py::reinterpret_borrow<py::object>(tiny->ptr(getPyInterpreter()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
}
