#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <filesystem>
#include <unistd.h>

#include "common/hardware/hw.h"
#include "common/swaglog.h"
#include "common/tests/native_test.h"
#include "json11/json11.hpp"

void test_swaglog() {
  setenv("OPENPILOT_PREFIX", ("swaglog_test_" + std::to_string(getpid())).c_str(), 1);
  setenv("MANAGER_DAEMON", "swaglog_test", 1);
  setenv("DONGLE_ID", "test_dongle_id", 1);
  setenv("CLEAN", "1", 1);

  LOGD("native-cpp-log");
  const std::string root = Path::swaglog_ipc();
  const auto ready = std::filesystem::directory_iterator(root + "/ready");
  CHECK(std::distance(begin(ready), end(ready)) == 1);
  const auto filename = ready->path();
  std::ifstream file(filename, std::ios::binary);
  CHECK(file.good());
  const std::string buffer{std::istreambuf_iterator<char>(file), {}};
  file.close();
  CHECK(buffer.size() > 1);
  CHECK(buffer[0] == CLOUDLOG_DEBUG);
  std::string error;
  const auto message = json11::Json::parse(buffer.substr(1), error);
  CHECK(error.empty());
  CHECK(message["levelnum"].int_value() == CLOUDLOG_DEBUG);
  CHECK(message["msg"].string_value() == "native-cpp-log");
  CHECK(message["funcname"].string_value() == "test_swaglog");
  CHECK(message["filename"].string_value().find("test_swaglog.cc") != std::string::npos);
  CHECK(message["ctx"]["daemon"].string_value() == "swaglog_test");
  CHECK(message["ctx"]["dongle_id"].string_value() == "test_dongle_id");
  CHECK(message["ctx"]["dirty"].bool_value() == false);

  CHECK(unlink(filename.c_str()) == 0);
  CHECK(rmdir((root + "/pending").c_str()) == 0);  // Publishing leaves no partial file.
  CHECK(rmdir((root + "/ready").c_str()) == 0);
  CHECK(rmdir(root.c_str()) == 0);
}

int main(int argc, char **argv) {
  // Used by test_logmessaged.py to exercise the real C++ producer with Python's reader.
  if (argc >= 2 && std::string(argv[1]) == "--emit") {
    const std::string message{std::istreambuf_iterator<char>(std::cin), {}};
    for (int i = 0; i < (argc == 3 ? std::stoi(argv[2]) : 1); ++i) LOGI("%s", message.c_str());
    return 0;
  }
  return run_native_test(test_swaglog);
}
