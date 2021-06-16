
#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <climits>
#include <random>
#include <string>

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"

std::string random_bytes(int size) {
  std::random_device rd;
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());
  std::string bytes(size + 1, '\0');
  std::generate(bytes.begin(), bytes.end(), std::ref(rbe));
  return bytes;
}

TEST_CASE("util::read_file") {
  SECTION("read /proc") {
    std::string ret = util::read_file("/proc/self/cmdline");
    REQUIRE(ret.find("test_util") != std::string::npos);
  }
  SECTION("read file") {
    char filename[] = "/tmp/test_read_XXXXXX";
    int fd = mkstemp(filename);

    REQUIRE(util::read_file(filename).empty());

    std::string content = random_bytes(64 * 1024);
    write(fd, content.c_str(), content.size());
    std::string ret = util::read_file(filename);
    REQUIRE(ret == content);
    close(fd);
  }
  SECTION("read directory") {
    REQUIRE(util::read_file(".").empty());
  }
  SECTION("read non-existent file") {
    std::string ret = util::read_file("does_not_exist");
    REQUIRE(ret.empty());
  }
  SECTION("read non-permission") {
    REQUIRE(util::read_file("/proc/kmsg").empty());
  }
}

TEST_CASE("write params") {
  Params params;
  std::pair<std::string, std::string> data[] = {
      {"DongleId", "cb38263377b873ee"},
      {"AthenadPid", "123"},
      {"CarParams", "test"},
  };
  SECTION("sync write") {
    for (auto &[key, val] : data) {
      params.put(key, val);
      REQUIRE(params.get(key) == val);
    }
  }
  SECTION("async write") {
    for (auto &[key, val] : data) {
      params.asyncPut(key, val);
    }
    while (params.asyncIsWriting()) {
      util::sleep_for(20);
    }
    for (auto &[key, val] : data) {
      REQUIRE(params.get(key) == val);
    }
  }
}
