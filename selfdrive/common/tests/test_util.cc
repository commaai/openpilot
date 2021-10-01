
#include <dirent.h>
#include <sys/stat.h>
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
  SECTION("read /proc/version") {
    std::string ret = util::read_file("/proc/version");
    REQUIRE(ret.find("Linux version") != std::string::npos);
  }
  SECTION("read from sysfs") {
    std::string ret = util::read_file("/sys/power/wakeup_count");
    REQUIRE(!ret.empty());
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

TEST_CASE("util::file_exists") {
  char filename[] = "/tmp/test_file_exists_XXXXXX";
  int fd = mkstemp(filename);
  REQUIRE(fd != -1);
  close(fd);

  SECTION("existent file") {
    REQUIRE(util::file_exists(filename));
    REQUIRE(util::file_exists("/tmp"));
  }
  SECTION("nonexistent file") {
    std::string fn = filename;
    REQUIRE(!util::file_exists(fn + "/nonexistent"));
  }
  SECTION("file has no access permissions") {
    std::string fn = "/proc/kmsg";
    std::ifstream f(fn);
    REQUIRE(f.good() == false);
    REQUIRE(util::file_exists(fn));
  }
  ::remove(filename);
}

TEST_CASE("util::read_files_in_dir") {
  char tmp_path[] = "/tmp/test_XXXXXX";
  const std::string test_path = mkdtemp(tmp_path);
  const std::string files[] = {".test1", "'test2'", "test3"};
  for (auto fn : files) {
    std::ofstream{test_path + "/" + fn} << fn;
  }
  mkdir((test_path + "/dir").c_str(), 0777);

  std::map<std::string, std::string> result = util::read_files_in_dir(test_path);
  REQUIRE(result.find("dir") == result.end());
  REQUIRE(result.size() == std::size(files));
  for (auto& [k, v] : result) {
    REQUIRE(k == v);
  }
}
