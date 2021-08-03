#include "selfdrive/proclogd/proclog.h"

#include <dirent.h>

#include <cassert>
#include <fstream>
#include <iterator>
#include <sstream>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

namespace Parser {

// parse /proc/stat
std::vector<CPUTime> cpuTimes(std::istream &iss) {
  std::vector<CPUTime> cpu_times;
  std::string line;
  // skip the first line for cpu total
  std::getline(iss, line);
  while (std::getline(iss, line)) {
    if (line.compare(0, 3, "cpu") != 0) break;

    CPUTime t = {};
    sscanf(line.data(), "cpu%d %lu %lu %lu %lu %lu %lu %lu",
            &t.id, &t.utime, &t.ntime, &t.stime, &t.itime, &t.iowtime, &t.irqtime, &t.sirqtime);
    cpu_times.push_back(t);
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

// field position (https://man7.org/linux/man-pages/man5/proc.5.html)
enum StatPos {
  pid = 1,
  state = 3,
  ppid = 4,
  utime = 14,
  stime = 15,
  cutime = 16,
  cstime = 17,
  priority = 18,
  nice = 19,
  num_threads = 20,
  starttime = 22,
  vsize = 23,
  rss = 24,
  processor = 39,
  MAX_FIELD = 52,
};

// parse /proc/pid/stat
std::optional<ProcStat> procStat(std::string stat) {
  // To avoid being fooled by names containing a closing paren, scan backwards.
  auto open_paren = stat.find('(');
  auto close_paren = stat.rfind(')');
  if (open_paren == std::string::npos || close_paren == std::string::npos || open_paren > close_paren)
    return std::nullopt;

  std::string name = stat.substr(open_paren + 1, close_paren - open_paren - 1);
  // repace space in name with _
  std::replace(&stat[open_paren], &stat[close_paren], ' ', '_');
  std::istringstream iss(stat);
  std::vector<std::string> v{std::istream_iterator<std::string>(iss),
                             std::istream_iterator<std::string>()};
  if (v.size() != StatPos::MAX_FIELD) {
    LOGE("failed to parse /proc/<pid>/stat :%s", stat.c_str());
    return std::nullopt;
  }

  try {
    ProcStat p = {};
    p.name = name;
    p.pid = stoi(v[StatPos::pid - 1]);
    p.state = v[StatPos::state - 1][0];
    p.ppid = stoi(v[StatPos::ppid - 1]);
    p.utime = stoul(v[StatPos::utime - 1]);
    p.stime = stoul(v[StatPos::stime - 1]);
    p.cutime = stol(v[StatPos::cutime - 1]);
    p.cstime = stol(v[StatPos::cstime - 1]);
    p.priority = stol(v[StatPos::priority - 1]);
    p.nice = stol(v[StatPos::nice - 1]);
    p.num_threads = stol(v[StatPos::num_threads - 1]);
    p.starttime = stoull(v[StatPos::starttime - 1]);
    p.vms = stoul(v[StatPos::vsize - 1]);
    p.rss = stoul(v[StatPos::rss - 1]);
    p.processor = stoi(v[StatPos::processor - 1]);
    return p;
  } catch (const std::invalid_argument &e) {
    LOGE("failed to parse /proc/<pid>/stat (invalid_argument) :%s", stat.c_str());
  } catch (const std::out_of_range &e) {
    LOGE("failed to parse /proc/<pid>/stat (out_of_range) :%s", stat.c_str());
  }
  return std::nullopt;
}

std::vector<int> pids() {
  std::vector<int> ids;
  DIR *d = opendir("/proc");
  assert(d);
  char *p_end;
  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (de->d_type == DT_DIR) {
      int pid = strtol(de->d_name, &p_end, 10);
      if (p_end == (de->d_name + strlen(de->d_name))) {
        ids.push_back(pid);
      }
    }
  }
  closedir(d);
  return ids;
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
    std::string spid = std::to_string(pid);
    cache.exe = util::readlink("/proc/" + spid + "/exe");
    cache.cmdline = cmdline(util::read_file("/proc/" + spid + "/cmdline"));
  }
  return cache;
}

}  // namespace Parser

const double jiffy = sysconf(_SC_CLK_TCK);
const size_t page_size = sysconf(_SC_PAGE_SIZE);

void buildCPUTimes(cereal::ProcLog::Builder &builder) {
  std::ifstream stream("/proc/stat");
  std::vector<CPUTime> stats = Parser::cpuTimes(stream);

  auto log_cpu_times = builder.initCpuTimes(stats.size());
  for (int i = 0; i < stats.size(); ++i) {
    auto l = log_cpu_times[i];
    const CPUTime &r = stats[i];
    l.setCpuNum(r.id);
    l.setUser(r.utime / jiffy);
    l.setNice(r.ntime / jiffy);
    l.setSystem(r.stime / jiffy);
    l.setIdle(r.itime / jiffy);
    l.setIowait(r.iowtime / jiffy);
    l.setIrq(r.irqtime / jiffy);
    l.setSoftirq(r.sirqtime / jiffy);
  }
}

void buildMemInfo(cereal::ProcLog::Builder &builder) {
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

void buildProcs(cereal::ProcLog::Builder &builder) {
  std::vector<ProcStat> proc_stats;
  for (int pid : Parser::pids()) {
    std::string path = "/proc/" + std::to_string(pid) + "/stat";
    if (auto stat = Parser::procStat(util::read_file(path))) {
      proc_stats.push_back(*stat);
    }
  }

  auto procs = builder.initProcs(proc_stats.size());
  for (size_t i = 0; i < proc_stats.size(); i++) {
    auto l = procs[i];
    const ProcStat &r = proc_stats[i];
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

void buildProcLogMessage(MessageBuilder &msg) {
  auto procLog = msg.initEvent().initProcLog();
  buildCPUTimes(procLog);
  buildMemInfo(procLog);
  buildProcs(procLog);
}
