#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/proclogd/parser.h"

TEST_CASE("Parser::procStat") {
  SECTION("from string") {
    std::string stat = "cpu 5 1 2 3 4 5 6 7 8\ncpu0 0 1 2 3 4 5 6 7 8\ncpu1 0 1 2 3 4 5 6 7 8";
    std::istringstream stream(stat);
    auto stats = Parser::procStat(stream);
    REQUIRE(stats.size() == 2);
    REQUIRE((stats[0].id == 0 && stats[0].utime == 0 && stats[0].ntime == 1));
    REQUIRE((stats[1].id == 1 && stats[0].utime == 0 && stats[0].ntime == 1));
  }
  SECTION("read from /proc/stat") {
    std::string stat = util::read_file("/proc/stat");
    std::istringstream stream(stat);
    auto stats = Parser::procStat(stream);
    REQUIRE(stats.size() == sysconf(_SC_NPROCESSORS_ONLN));
    for (int i = 0; i < stats.size(); ++i) {
      REQUIRE(stats[i].id == i);
    }
  }
}

TEST_CASE("Parser::memInfo") {
  SECTION("from string") {
    std::string stat = "MemTotal:    1024 kb\nMemFree:    10 kb\n";
    std::istringstream stream(stat);
    auto stats = Parser::memInfo(stream);
    REQUIRE((stats["MemTotal:"] = 1024 * 1024 && stats["MemFree:"] == 1024 * 10));
  }
  SECTION("from wrong string") {
    std::string stat = "MemTotal:   kb \nMemFree:    10 kb\n";
    std::istringstream stream(stat);
    auto stats = Parser::memInfo(stream);
    REQUIRE(stats.find("MemTotal:") == stats.end());
    REQUIRE(stats["MemFree:"] == 1024 * 10);
  }
  SECTION("read from /proc/ProcStat") {
    std::string stat = util::read_file("/proc/meminfo");
    std::istringstream stream(stat);
    auto stats = Parser::memInfo(stream);
    std::string keys[] = {"MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "Active:", "Inactive:", "Shmem:"};
    for (auto &key : keys) {
      REQUIRE(stats.find(key) != stats.end());
    }
  }
}

TEST_CASE("Parser::pidStat") {
  SECTION("normal") {
    std::string str = "167151 (cat) R 7297 167151 7297 34818 167151 4194304 92 0 0 0 0 0 0 0 20 0 1 0 303166 11563008 130 18446744073709551615 93994557333504 93994557359153 140732482410736 0 0 0 0 0 0 0 0 0 17 20 0 0 0 0 0 93994557377168 93994557378752 93994559090688 140732482412879 140732482412899 140732482412899 140732482416619 0";
    auto stat = Parser::pidStat(str);
    REQUIRE((stat && stat->name == "cat" && stat->pid == 167151));
  }
  SECTION("name with space") {
    std::string str = "167151 (cat 123) R 7297 167151 7297 34818 167151 4194304 92 0 0 0 0 0 0 0 20 0 1 0 303166 11563008 130 18446744073709551615 93994557333504 93994557359153 140732482410736 0 0 0 0 0 0 0 0 0 17 20 0 0 0 0 0 93994557377168 93994557378752 93994559090688 140732482412879 140732482412899 140732482412899 140732482416619 0";
    auto stat = Parser::pidStat(str);
    REQUIRE((stat && stat->name == "cat 123" && stat->pid == 167151));
  }
  SECTION("less") {
    std::string str = "167151 (cat 123) R 7297 167151";
    auto stat = Parser::pidStat(str);
    REQUIRE(!stat);
  }
  SECTION("more") {
    std::string str = "167151 (cat 123) R 7297 167151 7297 34818 167151 4194304 92 0 0 0 0 0 0 0 20 0 1 0 303166 11563008 130 18446744073709551615 93994557333504 93994557359153 140732482410736 0 0 0 0 0 0 0 0 0 17 20 0 0 0 0 0 93994557377168 93994557378752 93994559090688 140732482412879 140732482412899 140732482412899 140732482416619 0 1 2 3";
    auto stat = Parser::pidStat(str);
    REQUIRE(stat);
  }
}

TEST_CASE("Parser::cmdline") {
  SECTION("normal") {
    std::string str("a\0b\0c\0", 7);
    auto cmds = Parser::cmdline(str);
    REQUIRE(cmds.size() == 3);
    REQUIRE(cmds[0] == "a");
    REQUIRE(cmds[1] == "b");
    REQUIRE(cmds[2] == "c");
  }
  SECTION("multiple null") {
    std::string str("a\0\0\0b\0", 7);
    auto cmds = Parser::cmdline(str);
    REQUIRE(cmds.size() == 4);
    REQUIRE(cmds[0] == "a");
    REQUIRE(cmds[1] == "");
    REQUIRE(cmds[2] == "");
    REQUIRE(cmds[3] == "b");
  }
  SECTION("multiple null terminate") {
    std::string str("a\0b\0c\0\0\0", 9);
    auto cmds = Parser::cmdline(str);
    REQUIRE(cmds[0] == "a");
    REQUIRE(cmds[1] == "b");
    REQUIRE(cmds[2] == "c");
  }
}
