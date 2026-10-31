#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "common/hardware/hw.h"
#include "common/swaglog.h"
#include "common/tests/native_test.h"
#include "json11/json11.hpp"

void test_swaglog() {
  setenv("MANAGER_DAEMON", "swaglog_test", 1);
  setenv("DONGLE_ID", "test_dongle_id", 1);
  setenv("CLEAN", "1", 1);

  const std::string path = Path::swaglog_ipc();
  unlink(path.c_str());
  int sock = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(sock >= 0);
  struct timeval timeout = {.tv_sec = 5, .tv_usec = 0};
  CHECK(setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  CHECK(bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);

  LOGD("native-cpp-log");

  char buffer[4096] = {};
  const int size = recv(sock, buffer, sizeof(buffer), 0);
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

  CHECK(close(sock) == 0);
  CHECK(unlink(path.c_str()) == 0);
}

int main() {
  return run_native_test(test_swaglog);
}
