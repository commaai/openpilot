#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/proclogd/proclog.h"

const std::string allowed_states = "RSDTZtWXxKWPI";

TEST_CASE("Parser::pidStat") {
  std::string self_stat = util::read_file("/proc/self/stat");
  SECTION("self stat") {
    auto stat = Parser::procStat(self_stat);
    REQUIRE((stat && stat->name == "test_proclog" && stat->pid == getpid()));
    REQUIRE(stat->state == 'R');
  }
  SECTION("name with space and paren") {
    std::string from = "(test_proclog)";
    size_t start_pos = self_stat.find(from);
    self_stat.replace(start_pos, from.length(), "((test proclog))");
    auto stat = Parser::procStat(self_stat);
    REQUIRE(stat);
    REQUIRE(stat->name == "(test proclog)");
    REQUIRE(stat->pid == getpid());
  }
  SECTION("from empty string") {
    auto stat = Parser::procStat("");
    REQUIRE(!stat);
  }
  SECTION("all processes stats") {
    std::vector<int> pids = Parser::pids();
    REQUIRE(pids.size() > 1);

    std::set<int> parsed_pids;
    for (int pid : pids) {
      if (auto stat = Parser::procStat(util::read_file("/proc/" + std::to_string(pid) + "/stat"))) {
        REQUIRE(stat->pid == pid);
        REQUIRE(allowed_states.find(stat->state) != std::string::npos);
        parsed_pids.insert(pid);
      }
    }
    REQUIRE(parsed_pids.size() == pids.size());
  }
}

TEST_CASE("Parser::cpuTimes") {
  SECTION("from string") {
    std::string stat = "cpu 0 0 0 0 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8\ncpu1 1 2 3 4 5 6 7 8\nothers";
    std::istringstream stream(stat);
    auto stats = Parser::cpuTimes(stream);
    REQUIRE(stats.size() == 2);
    REQUIRE((stats[0].id == 0 && stats[0].utime == 1 && stats[0].ntime == 2));
    REQUIRE((stats[1].id == 1 && stats[0].utime == 1 && stats[0].ntime == 2));
  }
  SECTION("from /proc/stat") {
    std::istringstream stream(util::read_file("/proc/stat"));
    auto stats = Parser::cpuTimes(stream);
    REQUIRE(stats.size() == sysconf(_SC_NPROCESSORS_ONLN));
    for (int i = 0; i < stats.size(); ++i) {
      REQUIRE(stats[i].id == i);
    }
  }
  SECTION("from empty string") {
    std::istringstream stream("");
    REQUIRE(Parser::cpuTimes(stream).empty());
  }
}

TEST_CASE("Parser::memInfo") {
  SECTION("from string") {
    std::istringstream stream("MemTotal:    1024 kb\nMemFree:    2048 kb\n");
    auto stats = Parser::memInfo(stream);
    REQUIRE((stats["MemTotal:"] = 1024 * 1024 && stats["MemFree:"] == 2048 * 1024));
  }
  SECTION("from wrong string") {
    std::istringstream stream("MemTotal:   kb \nMemFree:    2048 kb\n");
    auto stats = Parser::memInfo(stream);
    REQUIRE(stats.find("MemTotal:") == stats.end());
    REQUIRE(stats["MemFree:"] == 2048 * 1024);
  }
  SECTION("from /proc/ProcStat") {
    std::istringstream stream(util::read_file("/proc/meminfo"));
    auto stats = Parser::memInfo(stream);
    std::string keys[] = {"MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "Active:", "Inactive:", "Shmem:"};
    for (auto &key : keys) {
      REQUIRE(stats.find(key) != stats.end());
      REQUIRE(stats[key] > 0);
    }
  }
  SECTION("from empty string") {
    std::istringstream stream("");
    auto stats = Parser::memInfo(stream);
    REQUIRE(stats.empty());
  }
}

void test_cmdline(std::string cmdline, const std::vector<std::string> requires) {
  auto cmds = Parser::cmdline(cmdline);
  REQUIRE(cmds.size() == requires.size());
  for (int i = 0; i < requires.size(); ++i) {
    REQUIRE(cmds[i] == requires[i]);  
  }
}
TEST_CASE("Parser::cmdline") {
  SECTION("normal") {
    test_cmdline(std::string("a\0b\0c\0", 7), {"a", "b", "c"});
  }
  SECTION("multiple null") {
    test_cmdline(std::string("a\0\0c\0", 6), {"a", "", "c"});
  }
  SECTION("multiple null terminate") {
    test_cmdline(std::string("a\0b\0c\0\0\0", 9), {"a", "b", "c"});
  }
  SECTION("from empty string") {
    auto cmds = Parser::cmdline("");
    REQUIRE(cmds.empty());
  }
}

TEST_CASE("buildProcLogerMessage") {
  MessageBuilder msg;
  buildProcLogMessage(msg);

  kj::Array<capnp::word> buf = capnp::messageToFlatArray(msg);
  capnp::FlatArrayMessageReader reader(buf);
  auto log = reader.getRoot<cereal::Event>().getProcLog();
  REQUIRE(log.totalSize().wordCount > 0);

  // test cereal::ProcLog::CPUTimes
  auto cpu_times = log.getCpuTimes();
  REQUIRE(cpu_times.size() == sysconf(_SC_NPROCESSORS_ONLN));
  REQUIRE(cpu_times[cpu_times.size() - 1].getCpuNum() == cpu_times.size() - 1);

  // test cereal::ProcLog::Mem
  auto mem = log.getMem();
  // first & last items we read from /proc/mem
  REQUIRE(mem.getTotal() > 0);
  REQUIRE(mem.getShared() > 0);

  // test cereal::ProcLog::Process
  auto procs = log.getProcs();
  REQUIRE(procs.size() > 1);

  bool found_self = false;
  int self_pid = getpid();
  for (auto p : procs) {
    if (p.getPid() == self_pid) {
      REQUIRE(p.getName() == "test_proclog");
      REQUIRE(p.getState() == 'R');
      found_self = true;
    }
    REQUIRE(p.getPid() != 0);
    REQUIRE(allowed_states.find(p.getState()) != std::string::npos);
  }
  REQUIRE(found_self == true);
}
