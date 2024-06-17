#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "common/util.h"
#include "system/proclogd/proclog.h"

const std::string allowed_states = "RSDTZtWXxKWPI";

TEST_CASE("Parser::procStat") {
  SECTION("from string") {
    const std::string stat_str =
        "33012 (code )) S 32978 6620 6620 0 -1 4194368 2042377 0 144 0 24510 11627 0 "
        "0 20 0 39 0 53077 830029824 62214 18446744073709551615 94257242783744 94257366235808 "
        "140735738643248 0 0 0 0 4098 1073808632 0 0 0 17 2 0 0 2 0 0 94257370858656 94257371248232 "
        "94257404952576 140735738648768 140735738648823 140735738648823 140735738650595 0";
    auto stat = Parser::procStat(stat_str);
    REQUIRE(stat);
    REQUIRE(stat->pid == 33012);
    REQUIRE(stat->name == "code )");
    REQUIRE(stat->state == 'S');
    REQUIRE(stat->ppid == 32978);
    REQUIRE(stat->utime == 24510);
    REQUIRE(stat->stime == 11627);
    REQUIRE(stat->cutime == 0);
    REQUIRE(stat->cstime == 0);
    REQUIRE(stat->priority == 20);
    REQUIRE(stat->nice == 0);
    REQUIRE(stat->num_threads == 39);
    REQUIRE(stat->starttime == 53077);
    REQUIRE(stat->vms == 830029824);
    REQUIRE(stat->rss == 62214);
    REQUIRE(stat->processor == 2);
  }
  SECTION("all processes") {
    std::vector<int> pids = Parser::pids();
    REQUIRE(pids.size() > 1);
    for (int pid : pids) {
      std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";
      INFO(stat_path);
      if (auto stat = Parser::procStat(util::read_file(stat_path))) {
        REQUIRE(stat->pid == pid);
        REQUIRE(allowed_states.find(stat->state) != std::string::npos);
      } else {
        REQUIRE(util::file_exists(stat_path) == false);
      }
    }
  }
}

TEST_CASE("Parser::cpuTimes") {
  SECTION("from string") {
    std::string stat =
        "cpu  0 0 0 0 0 0 0 0 0 0\n"
        "cpu0 1 2 3 4 5 6 7 8 9 10\n"
        "cpu1 1 2 3 4 5 6 7 8 9 10\n";
    std::istringstream stream(stat);
    auto stats = Parser::cpuTimes(stream);
    REQUIRE(stats.size() == 2);
    for (int i = 0; i < stats.size(); ++i) {
      REQUIRE(stats[i].id == i);
      REQUIRE(stats[i].utime == 1);
      REQUIRE(stats[i].ntime ==2);
      REQUIRE(stats[i].stime == 3);
      REQUIRE(stats[i].itime == 4);
      REQUIRE(stats[i].iowtime == 5);
      REQUIRE(stats[i].irqtime == 6);
      REQUIRE(stats[i].sirqtime == 7);
    }
  }
  SECTION("all cpus") {
    std::istringstream stream(util::read_file("/proc/stat"));
    auto stats = Parser::cpuTimes(stream);
    REQUIRE(stats.size() == sysconf(_SC_NPROCESSORS_ONLN));
    for (int i = 0; i < stats.size(); ++i) {
      REQUIRE(stats[i].id == i);
    }
  }
}

TEST_CASE("Parser::memInfo") {
  SECTION("from string") {
    std::istringstream stream("MemTotal:    1024 kb\nMemFree:    2048 kb\n");
    auto meminfo = Parser::memInfo(stream);
    REQUIRE(meminfo["MemTotal:"] == 1024 * 1024);
    REQUIRE(meminfo["MemFree:"] == 2048 * 1024);
  }
  SECTION("from /proc/meminfo") {
    std::string require_keys[] = {"MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "Active:", "Inactive:", "Shmem:"};
    std::istringstream stream(util::read_file("/proc/meminfo"));
    auto meminfo = Parser::memInfo(stream);
    for (auto &key : require_keys) {
      REQUIRE(meminfo.find(key) != meminfo.end());
      REQUIRE(meminfo[key] > 0);
    }
  }
}

void test_cmdline(std::string cmdline, const std::vector<std::string> requires) {
  std::stringstream ss;
  ss.write(&cmdline[0], cmdline.size());
  auto cmds = Parser::cmdline(ss);
  REQUIRE(cmds.size() == requires.size());
  for (int i = 0; i < requires.size(); ++i) {
    REQUIRE(cmds[i] == requires[i]);
  }
}
TEST_CASE("Parser::cmdline") {
  test_cmdline(std::string("a\0b\0c\0", 7), {"a", "b", "c"});
  test_cmdline(std::string("a\0\0c\0", 6), {"a", "c"});
  test_cmdline(std::string("a\0b\0c\0\0\0", 9), {"a", "b", "c"});
}

TEST_CASE("buildProcLoggerMessage") {
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
  REQUIRE(mem.getTotal() > 0);
  REQUIRE(mem.getShared() > 0);

  // test cereal::ProcLog::Process
  auto procs = log.getProcs();
  for (auto p : procs) {
    REQUIRE(allowed_states.find(p.getState()) != std::string::npos);
    if (p.getPid() == ::getpid()) {
      REQUIRE(p.getName() == "test_proclog");
      REQUIRE(p.getState() == 'R');
      REQUIRE_THAT(p.getExe().cStr(), Catch::Matchers::Contains("test_proclog"));
      REQUIRE_THAT(p.getCmdline()[0], Catch::Matchers::Contains("test_proclog"));
    }
  }
}
