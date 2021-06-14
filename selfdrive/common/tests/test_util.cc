
#include <sys/types.h>
#include <dirent.h>

#include <string>

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "selfdrive/common/util.h"

std::string random_string(size_t length) {
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

TEST_CASE("util::read_file") {
  auto read_file = [](const std::string &fn) -> std::string {
    std::ifstream f(fn);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
  };

  auto test_read_proc = [=]() {
    DIR *d = opendir("/proc");
    struct dirent *de = nullptr;
    while ((de = readdir(d))) {
      const std::string path = util::string_format("/proc/%s/cmdline", de->d_name);
      std::string ret1 = util::read_file(path);
      std::string ret2 = read_file(path);
      REQUIRE(ret1 == ret2);
    }
    closedir(d);
  };

  auto test_read_file = []() {
    std::vector<std::string> test_data;
    test_data.push_back(""); // test read empty file
    // continue reading to 64kb
    for (int i = 0; i < 64; ++i) {
      test_data.push_back(random_string(1024));
    }

    char filename[] = "/tmp/test_read_XXXXXX";
    int fd = mkstemp(filename);
    std::string file_content;
    for (auto data : test_data) {
      write(fd, data.c_str(), data.size());
      file_content += data;
      std::string ret = util::read_file(filename);
      REQUIRE(ret == file_content);
    }
    close(fd);
  };

  test_read_proc();
  test_read_file();
}
