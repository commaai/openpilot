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
#include <sstream>
#include <unordered_map>
#include <utility>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

ExitHandler do_exit;

struct ProcCache {
  int pid;
  std::string name;
  std::vector<std::string> cmdline;
  std::string exe;
};
double jiffy = sysconf(_SC_CLK_TCK);
size_t page_size = sysconf(_SC_PAGE_SIZE);

void buildCPUTimes(cereal::ProcLog::Builder &builder) {
  struct CPUTime {
    int id;
    unsigned long utime, ntime, stime, itime;
    unsigned long iowtime, irqtime, sirqtime;
  };
  std::vector<CPUTime> cpu_times;

  std::ifstream sstat("/proc/stat");
  std::string stat_line;
  while (std::getline(sstat, stat_line)) {
    if (util::starts_with(stat_line, "cpu ")) {
      // cpu total
    } else if (util::starts_with(stat_line, "cpu")) {
      // specific cpu
      CPUTime &t = cpu_times.emplace_back();
      sscanf(stat_line.data(), "cpu%d %lu %lu %lu %lu %lu %lu %lu",
             &t.id, &t.utime, &t.ntime, &t.stime, &t.itime, &t.iowtime, &t.irqtime, &t.sirqtime);
    } else {
      break;
    }
  }

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
  auto mem = builder.initMem();

  std::ifstream smem("/proc/meminfo");
  std::string mem_line;

  uint64_t mem_total = 0, mem_free = 0, mem_available = 0, mem_buffers = 0;
  uint64_t mem_cached = 0, mem_active = 0, mem_inactive = 0, mem_shared = 0;

  while (std::getline(smem, mem_line)) {
    if (util::starts_with(mem_line, "MemTotal:"))
      sscanf(mem_line.data(), "MemTotal: %" SCNu64 " kB", &mem_total);
    else if (util::starts_with(mem_line, "MemFree:"))
      sscanf(mem_line.data(), "MemFree: %" SCNu64 " kB", &mem_free);
    else if (util::starts_with(mem_line, "MemAvailable:"))
      sscanf(mem_line.data(), "MemAvailable: %" SCNu64 " kB", &mem_available);
    else if (util::starts_with(mem_line, "Buffers:"))
      sscanf(mem_line.data(), "Buffers: %" SCNu64 " kB", &mem_buffers);
    else if (util::starts_with(mem_line, "Cached:"))
      sscanf(mem_line.data(), "Cached: %" SCNu64 " kB", &mem_cached);
    else if (util::starts_with(mem_line, "Active:"))
      sscanf(mem_line.data(), "Active: %" SCNu64 " kB", &mem_active);
    else if (util::starts_with(mem_line, "Inactive:"))
      sscanf(mem_line.data(), "Inactive: %" SCNu64 " kB", &mem_inactive);
    else if (util::starts_with(mem_line, "Shmem:"))
      sscanf(mem_line.data(), "Shmem: %" SCNu64 " kB", &mem_shared);
  }

  mem.setTotal(mem_total * 1024);
  mem.setFree(mem_free * 1024);
  mem.setAvailable(mem_available * 1024);
  mem.setBuffers(mem_buffers * 1024);
  mem.setCached(mem_cached * 1024);
  mem.setActive(mem_active * 1024);
  mem.setInactive(mem_inactive * 1024);
  mem.setShared(mem_shared * 1024);
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

void buildProcesses(cereal::ProcLog::Builder &builder) {
  struct Process {
    int pid, ppid, processor;
    char state;
    unsigned long utime, stime;
    long cutime, cstime, priority, nice, num_threads;
    unsigned long long starttime;
    unsigned long vms, rss;
    std::string name;
  };
  std::vector<Process> procs_info;
  struct dirent *de = NULL;
  DIR *d = opendir("/proc");
  assert(d);
  while ((de = readdir(d))) {
    if (isdigit(de->d_name[0])) {
      Process p;
      p.pid = atoi(de->d_name);
      char name[PATH_MAX] = {};
      std::string stat = util::read_file(util::string_format("/proc/%d/stat", p.pid));
      int count = sscanf(stat.data(),
                         "%*d (%1024[^)]) %c %d %*d %*d %*d %*d %*d %*d %*d %*d %*d "
                         "%lu %lu %ld %ld %ld %ld %ld %*d %lld "
                         "%lu %lu %*d %*d %*d %*d %*d %*d %*d "
                         "%*d %*d %*d %*d %*d %*d %*d %d",
                         name, &p.state, &p.ppid,
                         &p.utime, &p.stime, &p.cutime, &p.cstime, &p.priority, &p.nice, &p.num_threads, &p.starttime,
                         &p.vms, &p.rss, &p.processor);
      if (count == 14) {
        p.name = name;
        procs_info.push_back(p);
      }
    }
  }
  closedir(d);

  auto lprocs = builder.initProcs(procs_info.size());
  for (size_t i = 0; i < procs_info.size(); i++) {
    auto l = lprocs[i];
    Process &r = procs_info[i];
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

void test_CPUTimes() {

}
int main() {
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
