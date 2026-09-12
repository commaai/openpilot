#include <cstdlib>
#include <string>

#include <zmq.h>

#include "common/hardware/hw.h"
#include "common/swaglog.h"
#include "common/tests/native_test.h"
#include "json11/json11.hpp"

void test_swaglog() {
  setenv("MANAGER_DAEMON", "swaglog_test", 1);
  setenv("DONGLE_ID", "test_dongle_id", 1);
  setenv("CLEAN", "1", 1);

  void *context = zmq_ctx_new();
  CHECK(context != nullptr);
  void *socket = zmq_socket(context, ZMQ_PULL);
  CHECK(socket != nullptr);
  int timeout = 5000;
  CHECK(zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
  CHECK(zmq_bind(socket, Path::swaglog_ipc().c_str()) == 0);

  LOGD("native-cpp-log");

  char buffer[4096] = {};
  const int size = zmq_recv(socket, buffer, sizeof(buffer), 0);
  CHECK(size > 1);
  CHECK(buffer[0] == CLOUDLOG_DEBUG);
  std::string error;
  const auto message = json11::Json::parse(std::string(buffer + 1, size - 1), error);
  CHECK(error.empty());
  CHECK(message["levelnum"].int_value() == CLOUDLOG_DEBUG);
  CHECK(message["msg"].string_value() == "native-cpp-log");
  CHECK(message["funcname"].string_value() == "test_swaglog");
  CHECK(message["filename"].string_value().find("test_swaglog.cc") != std::string::npos);
  CHECK(message["ctx"]["daemon"].string_value() == "swaglog_test");
  CHECK(message["ctx"]["dongle_id"].string_value() == "test_dongle_id");
  CHECK(message["ctx"]["dirty"].bool_value() == false);

  CHECK(zmq_close(socket) == 0);
  CHECK(zmq_ctx_destroy(context) == 0);
}

int main() {
  return run_native_test(test_swaglog);
}
