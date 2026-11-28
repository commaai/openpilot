#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>

#include "msgq/visionipc/visionbuf.h"
#include "msgq/visionipc/visionipc.h"
#include "msgq/visionipc/visionipc_client.h"
#include "msgq/visionipc/visionipc_server.h"

namespace nb = nanobind;

namespace {

struct PythonVisionIpcClient : VisionIpcClient {
  using VisionIpcClient::VisionIpcClient;
  VisionIpcBufExtra extra = {};
};

using Buffer = nb::ndarray<nb::memview, uint8_t, nb::ndim<1>, nb::device::cpu>;
using InputBuffer = nb::ndarray<const uint8_t, nb::ndim<1>, nb::device::cpu>;

}  // namespace

NB_MODULE(_visionipc, m) {
  // Daemon threads can still hold clients and servers at exit.
  nb::set_leak_warnings(false);

  m.def("get_endpoint_name", &get_endpoint_name, nb::arg("name"), nb::arg("stream"));

  nb::class_<VisionBuf>(m, "VisionBuf")
    .def_prop_ro("data", [](VisionBuf &buf) { return Buffer(buf.addr, {buf.len}); }, nb::rv_policy::reference_internal,
                 nb::sig("def data(self) -> memoryview"))
    .def_ro("width", &VisionBuf::width)
    .def_ro("height", &VisionBuf::height)
    .def_ro("stride", &VisionBuf::stride)
    .def_ro("uv_offset", &VisionBuf::uv_offset)
    .def_ro("idx", &VisionBuf::idx)
    .def_ro("fd", &VisionBuf::fd)
    .def_prop_ro("frame_id", &VisionBuf::get_frame_id);

  nb::class_<VisionIpcServer>(m, "VisionIpcServer")
    .def(nb::init<std::string>(), nb::arg("name"))
    .def("create_buffers", &VisionIpcServer::create_buffers, nb::arg("tp"), nb::arg("num_buffers"), nb::arg("width"), nb::arg("height"))
    .def("create_buffers_with_sizes", &VisionIpcServer::create_buffers_with_sizes,
         nb::arg("tp"), nb::arg("num_buffers"), nb::arg("width"), nb::arg("height"), nb::arg("size"), nb::arg("stride"), nb::arg("uv_offset"))
    .def("send", [](VisionIpcServer &server, VisionStreamType tp, InputBuffer data,
                    uint32_t frame_id, uint64_t timestamp_sof, uint64_t timestamp_eof) {
      VisionBuf *buf = server.get_buffer(tp);
      if (buf->len != data.size()) {
        PyErr_SetString(PyExc_AssertionError, "buffer size does not match frame size");
        throw nb::python_error();
      }
      // The old binding accepted strided byte views as well as contiguous data.
      auto *source = data.data();
      if (data.stride(0) == 1) {
        std::memcpy(buf->addr, source, data.size());
      } else {
        auto *target = static_cast<uint8_t *>(buf->addr);
        for (size_t i = 0; i < data.size(); ++i) target[i] = source[static_cast<int64_t>(i) * data.stride(0)];
      }
      buf->set_frame_id(frame_id);
      VisionIpcBufExtra extra = {frame_id, timestamp_sof, timestamp_eof, false};
      server.send(buf, &extra, false);
    }, nb::arg("tp"), nb::arg("data").noconvert(), nb::arg("frame_id") = 0, nb::arg("timestamp_sof") = 0, nb::arg("timestamp_eof") = 0,
       nb::sig("def send(self, tp: int, data: bytes | bytearray | memoryview, frame_id: int = 0, timestamp_sof: int = 0, timestamp_eof: int = 0) -> None"))
    .def("start_listener", &VisionIpcServer::start_listener);

  nb::class_<PythonVisionIpcClient>(m, "VisionIpcClient")
    .def(nb::init<std::string, VisionStreamType, bool>(), nb::arg("name"), nb::arg("stream"), nb::arg("conflate"))
    .def_prop_ro("width", [](PythonVisionIpcClient &client) -> std::optional<size_t> {
      return client.num_buffers ? std::make_optional(client.buffers[0].width) : std::nullopt;
    })
    .def_prop_ro("height", [](PythonVisionIpcClient &client) -> std::optional<size_t> {
      return client.num_buffers ? std::make_optional(client.buffers[0].height) : std::nullopt;
    })
    .def_prop_ro("stride", [](PythonVisionIpcClient &client) -> std::optional<size_t> {
      return client.num_buffers ? std::make_optional(client.buffers[0].stride) : std::nullopt;
    })
    .def_prop_ro("uv_offset", [](PythonVisionIpcClient &client) -> std::optional<size_t> {
      return client.num_buffers ? std::make_optional(client.buffers[0].uv_offset) : std::nullopt;
    })
    .def_prop_ro("buffer_len", [](PythonVisionIpcClient &client) -> std::optional<size_t> {
      return client.num_buffers ? std::make_optional(client.buffers[0].len) : std::nullopt;
    })
    .def_prop_ro("num_buffers", [](PythonVisionIpcClient &client) { return client.num_buffers; })
    .def_prop_ro("frame_id", [](PythonVisionIpcClient &client) { return client.extra.frame_id; })
    .def_prop_ro("timestamp_sof", [](PythonVisionIpcClient &client) { return client.extra.timestamp_sof; })
    .def_prop_ro("timestamp_eof", [](PythonVisionIpcClient &client) { return client.extra.timestamp_eof; })
    .def_prop_ro("valid", [](PythonVisionIpcClient &client) { return client.extra.valid; })
    .def("recv", [](PythonVisionIpcClient &client, int timeout_ms) {
      nb::gil_scoped_release release;
      return client.recv(&client.extra, timeout_ms);
    }, nb::arg("timeout_ms") = 100, nb::rv_policy::reference_internal,
       nb::sig("def recv(self, timeout_ms: int = 100) -> VisionBuf | None"))
    // Holds the GIL: connect remaps the buffers other threads may be reading.
    .def("connect", [](PythonVisionIpcClient &client, bool blocking) {
      return client.connect(blocking);
    }, nb::arg("blocking"))
    .def("is_connected", [](PythonVisionIpcClient &client) { return client.is_connected(); })
    .def_static("available_streams", &VisionIpcClient::getAvailableStreams, nb::arg("name"), nb::arg("block"),
                nb::call_guard<nb::gil_scoped_release>());
}
