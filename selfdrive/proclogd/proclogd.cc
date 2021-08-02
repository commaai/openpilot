#include <dirent.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

ExitHandler do_exit;

struct CPUTime {
  int id;
  unsigned long utime, ntime, stime, itime;
  unsigned long iowtime, irqtime, sirqtime;
};

struct PidStat {
    int pid, ppid, processor;
    char state;
    unsigned long utime, stime;
    long cutime, cstime, priority, nice, num_threads;
    unsigned long long starttime;
    unsigned long vms, rss;
    std::string name;
};

struct ProcCache {
  int pid;
  std::string name;
  std::vector<std::string> cmdline;
  std::string exe;
};

const double jiffy = sysconf(_SC_CLK_TCK);
const size_t page_size = sysconf(_SC_PAGE_SIZE);

// parse /proc/stat
std::vector<CPUTime> parseProcStat(std::istream &stream) {
  std::vector<CPUTime> cpu_times;
  std::string line;
  while (std::getline(stream, line)) {
    if (util::starts_with(line, "cpu ")) {
      // cpu total
    } else if (util::starts_with(line, "cpu")) {
      // specific cpu
      CPUTime t = {};
      sscanf(line.data(), "cpu%d %lu %lu %lu %lu %lu %lu %lu",
             &t.id, &t.utime, &t.ntime, &t.stime, &t.itime, &t.iowtime, &t.irqtime, &t.sirqtime);
      cpu_times.push_back(t);
    } else {
      break;
    }
  }
  return cpu_times;
}

// parse /proc/meminfo
std::unordered_map<std::string, uint64_t> parseMemInfo(std::istream &stream) {
  std::unordered_map<std::string, uint64_t> mem_info;
  std::string line, key;
  while (std::getline(stream, line)) {
    std::istringstream l(line);
    uint64_t val = 0;
    l >> key >> val;
    if (!l.fail()) {
      mem_info[key] = val * 1024;
    }
  }
  return mem_info;
}


const ProcCache &getProcExtraInfo(int pid, const std::string &name) {
  static std::unordered_map<pid_t, ProcCache> proc_cache;
  ProcCache &cache = proc_cache[pid];
  if (cache.pid != pid || cache.name != name) {
    cache.pid = pid;
    cache.name = name;
    cache.exe = util::readlink(util::string_format("/proc/%d/exe", pid));

    // null-delimited cmdline arguments to vector
    std::string cmdline_s = util::read_file(util::string_format("/proc/%d/cmdline", pid));
    const char *cmdline_p = cmdline_s.c_str();
    const char *cmdline_ep = cmdline_p + cmdline_s.size();

    // strip trailing null bytes
    while ((cmdline_ep - 1) > cmdline_p && *(cmdline_ep - 1) == 0) {
      cmdline_ep--;
    }

    cache.cmdline.clear();
    while (cmdline_p < cmdline_ep) {
      std::string arg(cmdline_p);
      cache.cmdline.push_back(arg);
      cmdline_p += arg.size() + 1;
    }
  }
  return cache;
}

// parse /proc/pid/stat
std::optional<PidStat> parsePidStat(const std::string &stat) {
  PidStat p;
  char name[PATH_MAX] = {};
  int count = sscanf(stat.data(),
                     "%d (%1024[^)]) %c %d %*d %*d %*d %*d %*d %*d %*d %*d %*d "
                     "%lu %lu %ld %ld %ld %ld %ld %*d %lld "
                     "%lu %lu %*d %*d %*d %*d %*d %*d %*d "
                     "%*d %*d %*d %*d %*d %*d %*d %d",
                     &p.pid, name, &p.state, &p.ppid,
                     &p.utime, &p.stime, &p.cutime, &p.cstime, &p.priority, &p.nice, &p.num_threads, &p.starttime,
                     &p.vms, &p.rss, &p.processor);
  if (count == 15) {
    p.name = name;
    return p;
  }
  return std::nullopt;
}

namespace Test {

void test_parseProcStat() {
  {
    std::string stat = "cpu 5 1 2 3 4 5 6 7 8\ncpu0 0 1 2 3 4 5 6 7 8\ncpu1 0 1 2 3 4 5 6 7 8";
    std::istringstream stream(stat);
    auto stats = parseProcStat(stream);
    assert(stats.size() == 2);
    assert(stats[0].id == 0 && stats[0].utime == 0 && stats[0].ntime == 1);
    assert(stats[1].id == 1 && stats[0].utime == 0 && stats[0].ntime == 1);
  }
  {
    std::string stat = util::read_file("/proc/stat");
    std::istringstream stream(stat);
    auto stats = parseProcStat(stream);
    assert(stats.size() == sysconf(_SC_NPROCESSORS_ONLN));
    for (int i = 0; i < stats.size(); ++i) {
      assert(stats[i].id == i);
    }
  }
}

void test_parseMemInfo() {
  {
    std::string stat = "MemTotal:    1024 kb\nMemFree:    10 kb\n";
    std::istringstream stream(stat);
    auto stats = parseMemInfo(stream);
    assert(stats["MemTotal:"] = 1024 * 1024 && stats["MemFree:"] == 1024 * 10);
  }
  {
    std::string stat = "MemTotal:   kb \nMemFree:    10 kb\n";
    std::istringstream stream(stat);
    auto stats = parseMemInfo(stream);
    assert(stats.find("MemTotal:") == stats.end());
    assert(stats["MemFree:"] == 1024 * 10);
  }
  {
    std::string stat = util::read_file("/proc/meminfo");
    std::istringstream stream(stat);
    auto stats = parseMemInfo(stream);
    std::string keys[] = {"MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "Active:", "Inactive:", "Shmem:"};
    for (auto &key : keys) {
      assert(stats.find(key) != stats.end());
    }
  }
}

void test_parsePidStat() {
  std::string normal = "167151 (cat) R 7297 167151 7297 34818 167151 4194304 92 0 0 0 0 0 0 0 20 0 1 0 303166 11563008 130 18446744073709551615 93994557333504 93994557359153 140732482410736 0 0 0 0 0 0 0 0 0 17 20 0 0 0 0 0 93994557377168 93994557378752 93994559090688 140732482412879 140732482412899 140732482412899 140732482416619 0";
  auto stat = parsePidStat(normal);
  assert(stat && stat->name == "cat" && stat->pid == 167151);

  std::string with_space = "167151 (cat 123) R 7297 167151 7297 34818 167151 4194304 92 0 0 0 0 0 0 0 20 0 1 0 303166 11563008 130 18446744073709551615 93994557333504 93994557359153 140732482410736 0 0 0 0 0 0 0 0 0 17 20 0 0 0 0 0 93994557377168 93994557378752 93994559090688 140732482412879 140732482412899 140732482412899 140732482416619 0";
  stat = parsePidStat(with_space);
  assert(stat && stat->name == "cat 123" && stat->pid == 167151);

  std::string less = "167151 (cat 123) R 7297 167151";
  stat = parsePidStat(less);
  assert(!stat);

  std::string more = "167151 (cat 123) R 7297 167151 7297 34818 167151 4194304 92 0 0 0 0 0 0 0 20 0 1 0 303166 11563008 130 18446744073709551615 93994557333504 93994557359153 140732482410736 0 0 0 0 0 0 0 0 0 17 20 0 0 0 0 0 93994557377168 93994557378752 93994559090688 140732482412879 140732482412899 140732482412899 140732482416619 0 1 2 3";
  stat = parsePidStat(less);
  assert(!stat);
}

int test_Parse() {
  Test::test_parseProcStat();
  Test::test_parseMemInfo();
  Test::test_parsePidStat();
  std::cout << "test ok" << std::endl;
  return 0;
}

}  // namespace Test

void buildCPUTimes(cereal::ProcLog::Builder &builder) {
  std::vector<CPUTime> cpu_times;

  std::ifstream stream("/proc/stat");
  std::vector<CPUTime> stats = parseProcStat(stream);

  auto log_cpu_times = builder.initCpuTimes(cpu_times.size());
  for (int i = 0; i < cpu_times.size(); ++i) {
    auto l = log_cpu_times[i];
    auto r = cpu_times[i];
    l.setCpuNum(r.id);
    l.setUser(r.utime / jiffy);
    l.setNice(r.ntime / jiffy);
    l.setSystem(r.stime / jiffy);
    l.setIdle(r.itime / jiffy);
    l.setIowait(r.iowtime / jiffy);
    l.setIrq(r.irqtime / jiffy);
    l.setSoftirq(r.irqtime / jiffy);
  }
}

void buildMemoryInfo(cereal::ProcLog::Builder &builder) {
  std::ifstream smem("/proc/meminfo");
  auto mem_info = parseMemInfo(smem);

  auto mem = builder.initMem();
  mem.setTotal(mem_info["MemTotal:"]);
  mem.setFree(mem_info["MemFree:"]);
  mem.setAvailable(mem_info["MemAvailable:"]);
  mem.setBuffers(mem_info["Buffers:"]);
  mem.setCached(mem_info["Cached:"]);
  mem.setActive(mem_info["Active:"]);
  mem.setInactive(mem_info["Inactive:"]);
  mem.setShared(mem_info["Shmem:"]);
}

void buildProcesses(cereal::ProcLog::Builder &builder) {
  std::vector<PidStat> procs_info;
  struct dirent *de = NULL;
  DIR *d = opendir("/proc");
  assert(d);
  while ((de = readdir(d))) {
    if (isdigit(de->d_name[0])) {
      int pid = atoi(de->d_name);
      auto stat = parsePidStat(util::read_file(util::string_format("/proc/%d/stat", pid)));
      if (stat) {
        procs_info.push_back(*stat);
      }
    }
  }
  closedir(d);

  auto lprocs = builder.initProcs(procs_info.size());
  for (size_t i = 0; i < procs_info.size(); i++) {
    auto l = lprocs[i];
    PidStat &r = procs_info[i];
    l.setPid(r.pid);
    l.setState(r.state);
    l.setPpid(r.ppid);
    l.setCpuUser(r.utime / jiffy);
    l.setCpuSystem(r.stime / jiffy);
    l.setCpuChildrenUser(r.cutime / jiffy);
    l.setCpuChildrenSystem(r.cstime / jiffy);
    l.setPriority(r.priority);
    l.setNice(r.nice);
    l.setNumThreads(r.num_threads);
    l.setStartTime(r.starttime / jiffy);
    l.setMemVms(r.vms);
    l.setMemRss((uint64_t)r.rss * page_size);
    l.setProcessor(r.processor);
    l.setName(r.name);

    const ProcCache &extra_info = getProcExtraInfo(r.pid, r.name);
    l.setExe(extra_info.exe);

    auto lcmdline = l.initCmdline(extra_info.cmdline.size());
    for (size_t i = 0; i < lcmdline.size(); i++) {
      lcmdline.set(i, extra_info.cmdline[i]);
    }
  }
}

int main(int argc, char** argv) {
  if (argc == 2 && strcmp(argv[1], "--test") == 0) {
    return Test::test_Parse();
  }
  setpriority(PRIO_PROCESS, 0, -15);
  PubMaster publisher({"procLog"});

  while (!do_exit) {
    MessageBuilder msg;
    auto procLog = msg.initEvent().initProcLog();
    buildCPUTimes(procLog);
    buildMemoryInfo(procLog);
    buildProcesses(procLog);
    publisher.send("procLog", msg);

    util::sleep_for(2000);  // 2 secs
  }

  return 0;
}
