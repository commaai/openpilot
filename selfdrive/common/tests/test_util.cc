
#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <climits>
#include <random>
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

template <typename OutputIt>
void random_bytes(OutputIt first, OutputIt last) {
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe;
  std::generate(first, last, std::ref(rbe));
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
  SECTION("read ascii file") {
    std::vector<std::string> test_data;
    test_data.push_back("");  // test read empty file
    // continue reading to 64kb
    for (int i = 0; i < 64; ++i) {
      test_data.push_back(random_string(1024));
    }

    char filename[] = "/tmp/test_read_XXXXXX";
    int fd = mkstemp(filename);
    std::string file_content;
    for (auto &data : test_data) {
      write(fd, data.c_str(), data.size());
      file_content += data;
      std::string ret = util::read_file(filename);
      REQUIRE(ret == file_content);
    }
    close(fd);
  }
  SECTION("read binary file") {
    std::vector<std::vector<uint8_t>> test_data;
    for (int i = 0; i < 64; ++i) {
      auto &dat = test_data.emplace_back();
      dat.resize(1024);
      random_bytes(dat.begin(), dat.end());
    }

    char filename[] = "/tmp/test_read_XXXXXX";
    int fd = mkstemp(filename);
    std::vector<uint8_t> file_content;
    for (auto &data : test_data) {
      write(fd, data.data(), data.size() * sizeof(uint8_t));
      file_content.insert(file_content.end(), data.begin(), data.end());
      std::string ret = util::read_file(filename);
      REQUIRE(memcmp(ret.data(), file_content.data(), file_content.size() * sizeof(uint8_t)) == 0);
    }
    close(fd);
  }
  SECTION("read non-exits file") {
    for (int i = 0; i < 5; ++i) {
      std::string ret = util::read_file(random_string(5));
      REQUIRE(ret == "");
    }
  }
}
