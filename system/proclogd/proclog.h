#include <optional>
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
  std::string name, exe;
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

std::vector<int> pids();
std::optional<ProcStat> procStat(std::string stat);
std::vector<std::string> cmdline(std::istream &stream);
std::vector<CPUTime> cpuTimes(std::istream &stream);
std::unordered_map<std::string, uint64_t> memInfo(std::istream &stream);
const ProcCache &getProcExtraInfo(int pid, const std::string &name);

};  // namespace Parser

void buildProcLogMessage(MessageBuilder &msg);
