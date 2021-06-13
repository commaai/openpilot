
#include <sys/types.h>
#include <dirent.h>

#include <string>

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "selfdrive/common/util.h"

TEST_CASE("util::read_file") {
  auto read_file = [](const std::string &fn) -> std::string {
    std::ifstream f(fn);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
  };

  auto read_directory = [=](const std::string &dir, const std::string &path_suffix = "") {
    DIR *d = opendir(dir.c_str());
    assert(d);
    struct dirent *de = nullptr;
    while ((de = readdir(d))) {
      if (!isalnum(de->d_name[0])) continue;

      const std::string path = util::string_format("%s/%s%s", dir.c_str(), de->d_name, path_suffix.c_str());
      std::string ret1 = util::read_file(path);
      std::string ret2 = read_file(path);
      REQUIRE(ret1 == ret2);
    }
    closedir(d);
  };

  read_directory("/proc", "/cmdline");
  read_directory(".");
}
