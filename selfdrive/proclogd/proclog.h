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
  std::string name;
  std::vector<std::string> cmdline;
  std::string exe;
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

namespace Parser {

std::vector<CPUTime> procStat(std::istream &stream);
std::unordered_map<std::string, uint64_t> memInfo(std::istream &stream);
std::optional<PidStat> pidStat(std::string stat);
std::vector<int> pids();
std::vector<std::string> cmdline(const std::string &cmd);
const ProcCache &getProcExtraInfo(int pid, const std::string &name);

};  // namespace Parser

void buildProcLogMessage(MessageBuilder &msg);
