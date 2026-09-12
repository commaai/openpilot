class LinuxSystemStats:
  def __init__(self) -> None:
    self._last_cpu_times = self._read_cpu_times()

  @staticmethod
  def _read_cpu_times() -> dict[int, tuple[int, int]]:
    cpu_times = {}
    with open('/proc/stat') as f:
      for line in f:
        name, *values = line.split()
        if not name.startswith('cpu') or not name[3:].isdigit():
          continue

        times = [int(value) for value in values]
        idle = sum(times[3:5])
        total = sum(times[:8])
        cpu_times[int(name[3:])] = (idle, total)
    return cpu_times

  def cpu_usage_percent(self) -> list[float]:
    current_cpu_times = self._read_cpu_times()
    usage = []
    for cpu, (idle, total) in sorted(current_cpu_times.items()):
      last_times = self._last_cpu_times.get(cpu)
      if last_times is None:
        usage.append(0.)
        continue

      last_idle, last_total = last_times
      idle_delta = idle - last_idle
      total_delta = total - last_total
      if idle_delta < 0 or total_delta <= 0:
        usage.append(0.)
      else:
        usage.append(max(0., min(100., 100. * (total_delta - idle_delta) / total_delta)))

    self._last_cpu_times = current_cpu_times
    return usage

  @staticmethod
  def memory_usage_percent() -> float:
    memory = {}
    with open('/proc/meminfo') as f:
      for line in f:
        key, value, *_ = line.split()
        if key in ('MemTotal:', 'MemAvailable:'):
          memory[key] = int(value)

    total = memory['MemTotal:']
    return max(0., min(100., 100. * (total - memory['MemAvailable:']) / total))
