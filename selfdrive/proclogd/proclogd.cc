#include <unistd.h>
#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cassert>
#include <memory>
#include <utility>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "messaging.hpp"

#include "common/timing.h"
#include "common/util.h"

ExitHandler do_exit;

namespace {
struct ProcCache {
  std::string name;
  std::vector<std::string> cmdline;
  std::string exe;
};

}

int main() {
  PubMaster publisher({"procLog"});

  double jiffy = sysconf(_SC_CLK_TCK);
  size_t page_size = sysconf(_SC_PAGE_SIZE);

  std::unordered_map<pid_t, ProcCache> proc_cache;

  while (!do_exit) {

    MessageBuilder msg;
    auto procLog = msg.initEvent().initProcLog();
    auto orphanage = msg.getOrphanage();

    // stat
    {
      std::vector<capnp::Orphan<cereal::ProcLog::CPUTimes>> otimes;

      std::ifstream sstat("/proc/stat");
      std::string stat_line;
      while (std::getline(sstat, stat_line)) {
        if (util::starts_with(stat_line, "cpu ")) {
          // cpu total
        } else if (util::starts_with(stat_line, "cpu")) {
          // specific cpu
          int id;
          unsigned long utime, ntime, stime, itime;
          unsigned long iowtime, irqtime, sirqtime;

          sscanf(stat_line.data(), "cpu%d %lu %lu %lu %lu %lu %lu %lu",
                 &id, &utime, &ntime, &stime, &itime, &iowtime, &irqtime, &sirqtime);

          auto ltimeo = orphanage.newOrphan<cereal::ProcLog::CPUTimes>();
          auto ltime = ltimeo.get();
          ltime.setCpuNum(id);
          ltime.setUser(utime / jiffy);
          ltime.setNice(ntime / jiffy);
          ltime.setSystem(stime / jiffy);
          ltime.setIdle(itime / jiffy);
          ltime.setIowait(iowtime / jiffy);
          ltime.setIrq(irqtime / jiffy);
          ltime.setSoftirq(irqtime / jiffy);

          otimes.push_back(std::move(ltimeo));

        } else {
          break;
        }
      }

      auto ltimes = procLog.initCpuTimes(otimes.size());
      for (size_t i = 0; i < otimes.size(); i++) {
        ltimes.adoptWithCaveats(i, std::move(otimes[i]));
      }
    }

    // meminfo
    {
      auto mem = procLog.initMem();

      std::ifstream smem("/proc/meminfo");
      std::string mem_line;

      uint64_t mem_total = 0, mem_free = 0, mem_available = 0, mem_buffers = 0;
      uint64_t mem_cached = 0, mem_active = 0, mem_inactive = 0, mem_shared = 0;

      while (std::getline(smem, mem_line)) {
        if (util::starts_with(mem_line, "MemTotal:")) sscanf(mem_line.data(), "MemTotal: %" SCNu64 " kB", &mem_total);
        else if (util::starts_with(mem_line, "MemFree:")) sscanf(mem_line.data(), "MemFree: %" SCNu64 " kB", &mem_free);
        else if (util::starts_with(mem_line, "MemAvailable:")) sscanf(mem_line.data(), "MemAvailable: %" SCNu64 " kB", &mem_available);
        else if (util::starts_with(mem_line, "Buffers:")) sscanf(mem_line.data(), "Buffers: %" SCNu64 " kB", &mem_buffers);
        else if (util::starts_with(mem_line, "Cached:")) sscanf(mem_line.data(), "Cached: %" SCNu64 " kB", &mem_cached);
        else if (util::starts_with(mem_line, "Active:")) sscanf(mem_line.data(), "Active: %" SCNu64 " kB", &mem_active);
        else if (util::starts_with(mem_line, "Inactive:")) sscanf(mem_line.data(), "Inactive: %" SCNu64 " kB", &mem_inactive);
        else if (util::starts_with(mem_line, "Shmem:")) sscanf(mem_line.data(), "Shmem: %" SCNu64 " kB", &mem_shared);
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

    // processes
    {
      std::vector<capnp::Orphan<cereal::ProcLog::Process>> oprocs;
      struct dirent *de = NULL;
      DIR *d = opendir("/proc");
      assert(d);
      while ((de = readdir(d))) {
        if (!isdigit(de->d_name[0])) continue;
        pid_t pid = atoi(de->d_name);


        auto lproco = orphanage.newOrphan<cereal::ProcLog::Process>();
        auto lproc = lproco.get();

        lproc.setPid(pid);

        char tcomm[PATH_MAX] = {0};

        {
          std::string stat = util::read_file(util::string_format("/proc/%d/stat", pid));

          char state;

          int ppid;
          unsigned long utime, stime;
          long cutime, cstime, priority, nice, num_threads;
          unsigned long long starttime;
          unsigned long vms, rss;
          int processor;

          int count = sscanf(stat.data(),
            "%*d (%1024[^)]) %c %d %*d %*d %*d %*d %*d %*d %*d %*d %*d "
             "%lu %lu %ld %ld %ld %ld %ld %*d %lld "
             "%lu %lu %*d %*d %*d %*d %*d %*d %*d "
             "%*d %*d %*d %*d %*d %*d %*d %d",
            tcomm, &state, &ppid,
            &utime, &stime, &cutime, &cstime, &priority, &nice, &num_threads, &starttime,
            &vms, &rss, &processor);

          if (count != 14) continue;

          lproc.setState(state);
          lproc.setPpid(ppid);
          lproc.setCpuUser(utime / jiffy);
          lproc.setCpuSystem(stime / jiffy);
          lproc.setCpuChildrenUser(cutime / jiffy);
          lproc.setCpuChildrenSystem(cstime / jiffy);
          lproc.setPriority(priority);
          lproc.setNice(nice);
          lproc.setNumThreads(num_threads);
          lproc.setStartTime(starttime / jiffy);
          lproc.setMemVms(vms);
          lproc.setMemRss((uint64_t)rss * page_size);
          lproc.setProcessor(processor);
        }

        std::string name(tcomm);
        lproc.setName(name);

        // populate other things from cache
        auto cache_it = proc_cache.find(pid);
        ProcCache cache;
        if (cache_it != proc_cache.end()) {
          cache = cache_it->second;
        }
        if (cache_it == proc_cache.end() || cache.name != name) {
          cache = (ProcCache){
            .name = name,
            .exe = util::readlink(util::string_format("/proc/%d/exe", pid)),
          };

          // null-delimited cmdline arguments to vector
          std::string cmdline_s = util::read_file(util::string_format("/proc/%d/cmdline", pid));
          const char* cmdline_p = cmdline_s.c_str();
          const char* cmdline_ep = cmdline_p + cmdline_s.size();

          // strip trailing null bytes
          while ((cmdline_ep-1) > cmdline_p && *(cmdline_ep-1) == 0) {
            cmdline_ep--;
          }

          while (cmdline_p < cmdline_ep) {
            std::string arg(cmdline_p);
            cache.cmdline.push_back(arg);
            cmdline_p += arg.size() + 1;
          }

          proc_cache[pid] = cache;
        }

        auto lcmdline = lproc.initCmdline(cache.cmdline.size());
        for (size_t i = 0; i < lcmdline.size(); i++) {
          lcmdline.set(i, cache.cmdline[i]);
        }
        lproc.setExe(cache.exe);

        oprocs.push_back(std::move(lproco));
      }
      closedir(d);

      auto lprocs = procLog.initProcs(oprocs.size());
      for (size_t i = 0; i < oprocs.size(); i++) {
        lprocs.adoptWithCaveats(i, std::move(oprocs[i]));
      }
    }

    publisher.send("procLog", msg);

    util::sleep_for(2000); // 2 secs
  }

  return 0;
}
