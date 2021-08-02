#include "selfdrive/proclogd/parser.h"

#include <cassert>
#include <climits>
#include <fstream>
#include <sstream>

#include "selfdrive/common/util.h"

// parse /proc/stat
std::vector<CPUTime> Parser::procStat(std::istream &stream) {
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
std::unordered_map<std::string, uint64_t> Parser::memInfo(std::istream &stream) {
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

// null-delimited cmdline arguments to vector
std::vector<std::string> Parser::cmdline(const std::string &cmdline_s) {
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

const ProcCache &Parser::getProcExtraInfo(int pid, const std::string &name) {
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

// parse /proc/pid/stat
std::optional<PidStat> Parser::pidStat(const std::string &stat) {
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
