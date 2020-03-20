def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


def print_cpu_usage(first_proc, last_proc):
  r = 0
  procs = [
      ("selfdrive.controls.controlsd", 59.46),
      ("./_modeld", 61.07),
      ("./loggerd", 28.49),
      ("selfdrive.controls.plannerd", 19.77),
      ("selfdrive.controls.radard", 9.54),
      ("./_ui", 9.54),
      ("./camerad", 7.07),
      ("selfdrive.locationd.locationd", 7.13),
      ("./_sensord", 6.17),
      ("selfdrive.controls.dmonitoringd", 5.48),
      ("./boardd", 3.63),
      ("./_dmonitoringmodeld", 2.67),
      ("selfdrive.logmessaged", 2.71),
      ("selfdrive.thermald", 2.41),
      ("./paramsd", 2.18),
      ("selfdrive.locationd.calibrationd", 1.76),
      ("./proclogd", 1.54),
      ("./_gpsd", 0.09),
      ("./clocksd", 0.02),
      ("./ubloxd", 0.02),
      ("selfdrive.tombstoned", 0),
      ("./logcatd", 0),
      ("selfdrive.updated", 0),
  ]

  dt = (last_proc.logMonoTime - first_proc.logMonoTime) / 1e9
  print("------------------------------------------------")
  for proc_name, normal_cpu_usage in procs:
    try:
      first = [p for p in first_proc.procLog.procs if proc_name in p.cmdline][0]
      last = [p for p in last_proc.procLog.procs if proc_name in p.cmdline][0]
      cpu_time = cputime_total(last) - cputime_total(first)
      cpu_usage = cpu_time / dt * 100.
      if cpu_usage > max(normal_cpu_usage * 1.1, normal_cpu_usage + 5.0):
        print(f"Warning {proc_name} using more CPU than normal")
        r = 1

      print(f"{proc_name.ljust(35)}  {cpu_usage:.2f}%")
    except IndexError:
      print(f"{proc_name.ljust(35)}  NO METRICS FOUND")
  print("------------------------------------------------")

  return r
