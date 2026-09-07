#include <cstdlib>
#include <string>

#include <cstdint>
#include <iostream>
#include <iterator>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <fcntl.h>
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

  // LOGD creates the queue and synchronously publishes a complete record.
  unlink(Path::swaglog_ipc().c_str());
  LOGD("native-cpp-log");

  int fd = open(Path::swaglog_ipc().c_str(), O_RDONLY);
  CHECK(fd >= 0);
  CHECK(flock(fd, LOCK_SH) == 0);
  void *mapping = mmap(nullptr, 4096, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(mapping != MAP_FAILED);
  const auto *positions = static_cast<const uint64_t *>(mapping);
  CHECK(positions[0] == 0);
  const auto *size_ptr = reinterpret_cast<const uint32_t *>(positions + 2);
  const int size = *size_ptr;
  CHECK(positions[1] == sizeof(uint32_t) + static_cast<size_t>(size));
  CHECK(size < 4096 - 20);
  const char *buffer = reinterpret_cast<const char *>(size_ptr + 1);
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

  // An initialized logger must drop instead of blocking on another descriptor's lock.
  const uint64_t write = positions[1];
  CHECK(flock(fd, LOCK_EX) == 0);
  LOGD("contended");
  CHECK(positions[1] == write);
  CHECK(flock(fd, LOCK_UN) == 0);
  LOGD("after contention");
  CHECK(positions[1] > write);

  CHECK(munmap(mapping, 4096) == 0);
  CHECK(close(fd) == 0);
  CHECK(unlink(Path::swaglog_ipc().c_str()) == 0);
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
