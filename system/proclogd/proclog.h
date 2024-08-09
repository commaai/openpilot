#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cereal/messaging/messaging.h"

struct CPUTime {
  int id;
  unsigned long utime, ntime, stime, itime;
  unsigned long iowtime, irqtime, sirqtime;
};

struct ProcCache {
  int pid;
  unsigned long start_time;
  std::string exe;
  std::vector<std::string> cmdline;
};

struct ProcStat {
  int pid, ppid, processor;
  char state;
  long cutime, cstime, priority, nice, num_threads, rss;
  unsigned long utime, stime, vms;
  unsigned long long starttime;
  std::string name;
};

namespace Parser {

std::set<int> pids();
std::optional<ProcStat> procStat(std::string stat);
std::vector<std::string> cmdline(std::istream &stream);
std::vector<CPUTime> cpuTimes(std::istream &stream);
std::unordered_map<std::string, uint64_t> memInfo(std::istream &stream);
ProcCache getProcExtraInfo(int pid, unsigned long start_time);

};  // namespace Parser

void buildProcLogMessage(MessageBuilder &msg);
