#include "system/proclogd/proclog.h"

#include <dirent.h>

#include <cassert>
#include <fstream>
#include <iterator>
#include <sstream>

#include "common/timing.h"
#include "common/swaglog.h"
#include "common/util.h"

namespace Parser {

// parse /proc/stat
std::vector<CPUTime> cpuTimes(std::istream &stream) {
  std::vector<CPUTime> cpu_times;
  std::string line;
  // skip the first line for cpu total
  std::getline(stream, line);
  while (std::getline(stream, line)) {
    if (line.compare(0, 3, "cpu") != 0) break;

    CPUTime t = {};
    std::istringstream iss(line);
    if (iss.ignore(3) >> t.id >> t.utime >> t.ntime >> t.stime >> t.itime >> t.iowtime >> t.irqtime >> t.sirqtime)
      cpu_times.push_back(t);
  }
  return cpu_times;
}

// parse /proc/meminfo
std::unordered_map<std::string, uint64_t> memInfo(std::istream &stream) {
  std::unordered_map<std::string, uint64_t> mem_info;
  std::string line, key;
  while (std::getline(stream, line)) {
    uint64_t val = 0;
    std::istringstream iss(line);
    if (iss >> key >> val) {
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
  if (open_paren == std::string::npos || close_paren == std::string::npos || open_paren > close_paren) {
    return std::nullopt;
  }

  std::string name = stat.substr(open_paren + 1, close_paren - open_paren - 1);
  // replace space in name with _
  std::replace(&stat[open_paren], &stat[close_paren], ' ', '_');
  std::istringstream iss(stat);
  std::vector<std::string> v{std::istream_iterator<std::string>(iss),
                             std::istream_iterator<std::string>()};
  try {
    if (v.size() != StatPos::MAX_FIELD) {
      throw std::invalid_argument("stat");
    }
    ProcStat p = {
      .name = name,
      .pid = stoi(v[StatPos::pid - 1]),
      .state = v[StatPos::state - 1][0],
      .ppid = stoi(v[StatPos::ppid - 1]),
      .utime = stoul(v[StatPos::utime - 1]),
      .stime = stoul(v[StatPos::stime - 1]),
      .cutime = stol(v[StatPos::cutime - 1]),
      .cstime = stol(v[StatPos::cstime - 1]),
      .priority = stol(v[StatPos::priority - 1]),
      .nice = stol(v[StatPos::nice - 1]),
      .num_threads = stol(v[StatPos::num_threads - 1]),
      .starttime = stoull(v[StatPos::starttime - 1]),
      .vms = stoul(v[StatPos::vsize - 1]),
      .rss = stol(v[StatPos::rss - 1]),
      .processor = stoi(v[StatPos::processor - 1]),
    };
    return p;
  } catch (const std::invalid_argument &e) {
    LOGE("failed to parse procStat (%s) :%s", e.what(), stat.c_str());
  } catch (const std::out_of_range &e) {
    LOGE("failed to parse procStat (%s) :%s", e.what(), stat.c_str());
  }
  return std::nullopt;
}

// return list of PIDs from /proc
std::set<int> pids() {
  std::set<int> ids;
  DIR *d = opendir("/proc");
  assert(d);
  char *p_end;
  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (de->d_type == DT_DIR) {
      int pid = strtol(de->d_name, &p_end, 10);
      if (p_end == (de->d_name + strlen(de->d_name))) {
        ids.insert(pid);
      }
    }
  }
  closedir(d);
  return ids;
}

// null-delimited cmdline arguments to vector
std::vector<std::string> cmdline(std::istream &stream) {
  std::vector<std::string> ret;
  std::string line;
  while (std::getline(stream, line, '\0')) {
    if (!line.empty()) {
      ret.push_back(line);
    }
  }
  return ret;
}

ProcCache getProcExtraInfo(int pid, unsigned long start_time) {
  std::string proc_path = "/proc/" + std::to_string(pid);
  std::ifstream stream(proc_path + "/cmdline");
  return {
    .pid = pid,
    .start_time = start_time,
    .exe = util::readlink(proc_path + "/exe"),
    .cmdline = cmdline(stream),
  };
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
  static std::unordered_map<pid_t, ProcCache> proc_cache;

  // Timestamp of the last cache cleanup
  static double last_cleanup_ts = millis_since_boot();
  double current_ts = millis_since_boot();

  // Get the current set of active process IDs
  auto pids = Parser::pids();

  // Remove stale entries from proc_cache if they no longer correspond to active PIDs
  if (current_ts - last_cleanup_ts > 30000) {  // 30 seconds
    for (auto it = proc_cache.begin(); it != proc_cache.end();) {
      it = pids.count(it->first) ? std::next(it) : proc_cache.erase(it);
    }
    last_cleanup_ts = current_ts;
  }

  // Collect process statistics for active PIDs
  std::vector<ProcStat> proc_stats;
  proc_stats.reserve(pids.size());
  for (int pid : pids) {
    std::string path = "/proc/" + std::to_string(pid) + "/stat";
    if (auto stat = Parser::procStat(util::read_file(path))) {
      proc_stats.push_back(*stat);
    }
  }

  // Build procs message
  auto procs = builder.initProcs(proc_stats.size());
  for (size_t i = 0; i < proc_stats.size(); i++) {
    const ProcStat &r = proc_stats[i];
    auto &extra_info = proc_cache[r.pid];

    // Update cache if the process is new or has restarted
    if (extra_info.pid != r.pid || extra_info.start_time != r.starttime) {
      extra_info = Parser::getProcExtraInfo(r.pid, r.starttime);
    }

    auto l = procs[i];
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
    l.setExe(extra_info.exe);

    auto lcmdline = l.initCmdline(extra_info.cmdline.size());
    for (size_t j = 0; j < lcmdline.size(); j++) {
      lcmdline.set(j, extra_info.cmdline[j]);
    }
  }
}

void buildProcLogMessage(MessageBuilder &msg) {
  auto procLog = msg.initEvent().initProcLog();
  buildProcs(procLog);
  buildCPUTimes(procLog);
  buildMemInfo(procLog);
}
