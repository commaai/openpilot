#include "selfdrive/proclogd/proclog.h"

#include <dirent.h>

#include <cassert>
#include <climits>
#include <fstream>
#include <sstream>

#include "selfdrive/common/util.h"

namespace Parser {

// parse /proc/stat
std::vector<CPUTime> procStat(std::istream &stream) {
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
std::unordered_map<std::string, uint64_t> memInfo(std::istream &stream) {
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

// parse /proc/pid/stat
std::optional<PidStat> pidStat(const std::string &stat) {
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

// null-delimited cmdline arguments to vector
std::vector<std::string> cmdline(const std::string &cmdline_s) {
  const char *cmdline_p = cmdline_s.c_str();
  const char *cmdline_ep = cmdline_p + cmdline_s.size();

  // strip trailing null bytes
  while ((cmdline_ep - 1) > cmdline_p && *(cmdline_ep - 1) == 0) {
    cmdline_ep--;
  }
  std::vector<std::string> ret;
  while (cmdline_p < cmdline_ep) {
    std::string arg(cmdline_p);
    ret.push_back(arg);
    cmdline_p += arg.size() + 1;
  }
  return ret;
}

const ProcCache &getProcExtraInfo(int pid, const std::string &name) {
  static std::unordered_map<pid_t, ProcCache> proc_cache;
  ProcCache &cache = proc_cache[pid];
  if (cache.pid != pid || cache.name != name) {
    cache.pid = pid;
    cache.name = name;
    cache.exe = util::readlink(util::string_format("/proc/%d/exe", pid));
    cache.cmdline = cmdline(util::read_file(util::string_format("/proc/%d/cmdline", pid)));
  }
  return cache;
}

}  // namespace Parser

const double jiffy = sysconf(_SC_CLK_TCK);
const size_t page_size = sysconf(_SC_PAGE_SIZE);

void buildCPUTimes(cereal::ProcLog::Builder &builder) {
  std::ifstream stream("/proc/stat");
  std::vector<CPUTime> stats = Parser::procStat(stream);

  auto log_cpu_times = builder.initCpuTimes(stats.size());
  for (int i = 0; i < stats.size(); ++i) {
    auto l = log_cpu_times[i];
    auto r = stats[i];
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
  std::ifstream stream("/proc/meminfo");
  auto mem_info = Parser::memInfo(stream);

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
      if (auto stat = Parser::pidStat(util::read_file(util::string_format("/proc/%d/stat", pid)))) {
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

    const ProcCache &extra_info = Parser::getProcExtraInfo(r.pid, r.name);
    l.setExe(extra_info.exe);
    auto lcmdline = l.initCmdline(extra_info.cmdline.size());
    for (size_t i = 0; i < lcmdline.size(); i++) {
      lcmdline.set(i, extra_info.cmdline[i]);
    }
  }
}

void buildProcLogerMessage(MessageBuilder &msg) {
  auto procLog = msg.initEvent().initProcLog();
  buildCPUTimes(procLog);
  buildMemoryInfo(procLog);
  buildProcesses(procLog);
}
