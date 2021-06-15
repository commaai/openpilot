
#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <climits>
#include <random>
#include <string>

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"

std::string random_bytes(int size) {
  std::random_device rd;
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());
  std::string bytes(size+1, '\0');
  std::generate(bytes.begin(), bytes.end(), std::ref(rbe));
  return bytes;
}

TEST_CASE("util::read_file") {
  auto read_file = [](const std::string &fn) -> std::string {
    std::ifstream f(fn);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
  };

  SECTION("read /proc") {
    DIR *d = opendir("/proc");
    struct dirent *de = nullptr;
    while ((de = readdir(d))) {
      const std::string path = util::string_format("/proc/%s/cmdline", de->d_name);
      std::string ret1 = util::read_file(path);
      std::string ret2 = read_file(path);
      REQUIRE(ret1 == ret2);
    }
    closedir(d);
  }
  SECTION("read file") {
    std::vector<std::string> test_data;
    test_data.push_back("");  // test read empty file
    for (int i = 0; i < 64; ++i) {
      test_data.push_back(random_bytes(1024));
    }

    char filename[] = "/tmp/test_read_XXXXXX";
    int fd = mkstemp(filename);
    std::string file_content;
    // continue writing&reading to 64kb
    for (auto &data : test_data) {
      write(fd, data.c_str(), data.size());
      file_content += data;
      std::string ret = util::read_file(filename);
      REQUIRE(ret == file_content);
    }
    close(fd);
  }
  SECTION("read non-existant file") {
    std::string ret = util::read_file("does_not_exist");
    REQUIRE(ret.empty());
  }
}
