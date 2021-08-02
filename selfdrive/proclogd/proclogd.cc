
#include <dirent.h>
#include <sys/resource.h>

#include <cassert>
#include <fstream>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/proclogd/parser.h"

ExitHandler do_exit;

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
      auto stat = Parser::pidStat(util::read_file(util::string_format("/proc/%d/stat", pid)));
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

    const ProcCache &extra_info = Parser::getProcExtraInfo(r.pid, r.name);
    l.setExe(extra_info.exe);

    auto lcmdline = l.initCmdline(extra_info.cmdline.size());
    for (size_t i = 0; i < lcmdline.size(); i++) {
      lcmdline.set(i, extra_info.cmdline[i]);
    }
  }
}

int main(int argc, char **argv) {
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
