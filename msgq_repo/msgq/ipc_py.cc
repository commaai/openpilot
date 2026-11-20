#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "msgq/event.h"
#include "msgq/ipc.h"

namespace nb = nanobind;

namespace {

struct IpcError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct MultiplePublishersError : IpcError {
  using IpcError::IpcError;
};

void check_error(bool failed, int error, const std::string &endpoint = "") {
  if (!failed) return;
  std::string message = "Messaging failure " + (endpoint.empty() ? "" : "with " + endpoint) + ": " + std::strerror(error);
  if (error == EADDRINUSE) throw MultiplePublishersError(message);
  throw IpcError(message);
}

}  // namespace

NB_MODULE(_ipc, m) {
  // Daemon threads can still hold sockets at exit.
  nb::set_leak_warnings(false);

  nb::exception<IpcError> ipc_error(m, "IpcError");
  nb::exception<MultiplePublishersError>(m, "MultiplePublishersError", ipc_error);

  m.def("toggle_fake_events", &SocketEventHandle::toggle_fake_events, nb::arg("enabled"));
  m.def("set_fake_prefix", &SocketEventHandle::set_fake_prefix, nb::arg("prefix"));
  m.def("get_fake_prefix", &SocketEventHandle::fake_prefix);
  m.def("delete_fake_prefix", [] { SocketEventHandle::set_fake_prefix(""); });
  m.def("wait_for_one_event", &Event::wait_for_one, nb::arg("events"), nb::arg("timeout") = -1, nb::call_guard<nb::gil_scoped_release>());

  nb::class_<Event>(m, "Event")
    .def(nb::init<>())
    .def("set", &Event::set)
    .def("clear", &Event::clear)
    .def("wait", &Event::wait, nb::arg("timeout") = -1, nb::call_guard<nb::gil_scoped_release>())
    .def("peek", &Event::peek)
    .def_prop_ro("fd", &Event::fd)
    .def_prop_ro("ptr", [](Event &event) { return reinterpret_cast<uintptr_t>(&event); });

  nb::class_<SocketEventHandle>(m, "SocketEventHandle")
    .def(nb::init<std::string, std::string, bool>(), nb::arg("endpoint"), nb::arg("identifier"), nb::arg("override"))
    .def_prop_rw("enabled", &SocketEventHandle::is_enabled, &SocketEventHandle::set_enabled)
    .def_prop_ro("recv_called_event", &SocketEventHandle::recv_called, nb::keep_alive<0, 1>())
    .def_prop_ro("recv_ready_event", &SocketEventHandle::recv_ready, nb::keep_alive<0, 1>());

  nb::class_<SubSocket>(m, "SubSocket")
    .def(nb::init<>())
    .def("connect", [](SubSocket &socket, const std::string &name, bool conflate, size_t segment_size) {
      nb::gil_scoped_release release;
      int result = socket.connect(name, conflate, segment_size);
      check_error(result != 0, errno, name);
    }, nb::arg("endpoint"), nb::arg("conflate") = false, nb::arg("segment_size") = 0)
    .def("setTimeout", &SubSocket::setTimeout, nb::arg("timeout"), nb::call_guard<nb::gil_scoped_release>())
    .def("receive", [](SubSocket &socket, bool non_blocking) -> std::optional<nb::bytes> {
      std::unique_ptr<Message> message;
      {
        nb::gil_scoped_release release;
        message.reset(socket.receive(non_blocking));
      }
      if (!message) return std::nullopt;
      return nb::bytes(message->getData(), message->getSize());
    }, nb::arg("non_blocking") = false);

  nb::class_<Poller>(m, "Poller")
    .def(nb::init<>())
    .def("registerSocket", &Poller::registerSocket, nb::arg("socket"), nb::keep_alive<1, 2>())
    .def("poll", &Poller::poll, nb::arg("timeout"), nb::rv_policy::reference, nb::call_guard<nb::gil_scoped_release>());

  nb::class_<PubSocket>(m, "PubSocket")
    .def(nb::init<>())
    .def("connect", [](PubSocket &socket, const std::string &name, size_t segment_size) {
      int result = socket.connect(name, segment_size);
      check_error(result != 0, errno, name);
    }, nb::arg("endpoint"), nb::arg("segment_size") = 0)
    .def("send", [](PubSocket &socket, nb::bytes data) {
      int result = socket.send(const_cast<char *>(data.c_str()), data.size());
      check_error(result != static_cast<int>(data.size()), errno);
    }, nb::arg("data"))
    .def("all_readers_updated", &PubSocket::all_readers_updated)
    .def("wait_for_readers", [](PubSocket &socket, double timeout, double interval) {
      // Python's sleep releases the GIL and handles signals while waiting.
      nb::object sleep = nb::module_::import_("time").attr("sleep");
      auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout);
      while (std::chrono::steady_clock::now() < deadline) {
        if (socket.all_readers_updated()) return;
        sleep(interval);
      }
      PyErr_SetString(PyExc_TimeoutError, "subscriber did not connect");
      throw nb::python_error();
    }, nb::arg("timeout") = 1.0, nb::arg("interval") = 0.001);
}
