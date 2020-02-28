def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


def print_cpu_usage(first_proc, last_proc):
    procs = [
        "selfdrive.thermald",
        "selfdrive.controls.controlsd",
        "selfdrive.controls.plannerd",
        "selfdrive.controls.radard",
        "selfdrive.controls.dmonitoringd",
        "./ubloxd",
        "./loggerd",
        "selfdrive.logmessaged",
        "selfdrive.locationd.locationd",
        "selfdrive.tombstoned",
        "./logcatd",
        "./proclogd",
        "./boardd",
        "./_ui",
        "selfdrive.locationd.calibrationd",
        "./paramsd",
        "./camerad",
        "./_sensord",
        "./clocksd",
        "./_gpsd",
        "selfdrive.updated",
        "./_dmonitoringmodeld",
        "./_modeld",
    ]

    dt = (last_proc.logMonoTime - first_proc.logMonoTime) / 1e9
    print("------------------------------------------------")
    for proc_name in procs:
      try:
        first = [p for p in first_proc.procLog.procs if proc_name in p.cmdline][0]
        last = [p for p in last_proc.procLog.procs if proc_name in p.cmdline][0]
        cpu_time = cputime_total(last) - cputime_total(first)
        cpu_usage = cpu_time / dt * 100.
        print(f"{proc_name.ljust(35)}  {cpu_usage:.2f}%")
      except IndexError:
        print(f"{proc_name.ljust(35)}  NO METRICS FOUND")
    print("------------------------------------------------")
