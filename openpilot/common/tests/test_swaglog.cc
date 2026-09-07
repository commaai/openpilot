#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <dirent.h>
#include <sys/wait.h>
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
  DIR *ready = opendir((root + "/ready").c_str());
  CHECK(ready != nullptr);
  std::string filename;
  int count = 0;
  while (const auto *entry = readdir(ready)) {
    if (entry->d_name[0] == '.') continue;
    filename = root + "/ready/" + entry->d_name;
    ++count;
  }
  CHECK(closedir(ready) == 0);
  CHECK(count == 1);
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
    if (argc == 3) {
      for (int i = 0; i < std::stoi(argv[2]); ++i) LOGD("%s:%d", message.c_str(), i);
    } else {
      LOGD("%s", message.c_str());
    }
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--fork") {
    LOGD("parent");
    const pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
      LOGD("child");
      _exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  }
  return run_native_test(test_swaglog);
}
