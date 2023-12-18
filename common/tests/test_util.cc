
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <climits>
#include <fstream>
#include <random>
#include <string>

#include "catch2/catch.hpp"
#include "common/util.h"

std::string random_bytes(int size) {
  std::random_device rd;
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());
  std::string bytes(size + 1, '\0');
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
    bool equal = (ret == content);
    REQUIRE(equal);
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


TEST_CASE("util::safe_fwrite") {
  char filename[] = "/tmp/XXXXXX";
  int fd = mkstemp(filename);
  close(fd);
  std::string dat = random_bytes(1024 * 1024);

  FILE *f = util::safe_fopen(filename, "wb");
  REQUIRE(f != nullptr);
  size_t size = util::safe_fwrite(dat.data(), 1, dat.size(), f);
  REQUIRE(size == dat.size());
  int ret = util::safe_fflush(f);
  REQUIRE(ret == 0);
  ret = fclose(f);
  REQUIRE(ret == 0);
  bool equal = (dat == util::read_file(filename));
  REQUIRE(equal);
}

TEST_CASE("util::create_directories") {
  system("rm /tmp/test_create_directories -rf");
  std::string dir = "/tmp/test_create_directories/a/b/c/d/e/f";

  auto check_dir_permissions = [](const std::string &dir, mode_t mode) -> bool {
    struct stat st = {};
    return stat(dir.c_str(), &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR && (st.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO)) == mode;
  };

  SECTION("create_directories") {
    REQUIRE(util::create_directories(dir, 0755));
    REQUIRE(check_dir_permissions(dir, 0755));
  }
  SECTION("dir already exists") {
    REQUIRE(util::create_directories(dir, 0755));
    REQUIRE(util::create_directories(dir, 0755));
  }
  SECTION("a file exists with the same name") {
    REQUIRE(util::create_directories(dir, 0755));
    int f = open((dir + "/file").c_str(), O_RDWR | O_CREAT);
    REQUIRE(f != -1);
    close(f);
    REQUIRE(util::create_directories(dir + "/file", 0755) == false);
    REQUIRE(util::create_directories(dir + "/file/1/2/3", 0755) == false);
  }
  SECTION("end with slashes") {
    REQUIRE(util::create_directories(dir + "/", 0755));
  }
  SECTION("empty") {
    REQUIRE(util::create_directories("", 0755) == false);
  }
}
