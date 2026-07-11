class SystemStats:
  def __init__(self) -> None:
    self._last_cpu_times = self._read_cpu_times()

  @staticmethod
  def _read_cpu_times() -> list[tuple[int, int]]:
    cpu_times = []
    with open('/proc/stat') as f:
      for line in f:
        name, *values = line.split()
        if not name.startswith('cpu') or not name[3:].isdigit():
          continue

        times = [int(value) for value in values]
        idle = sum(times[3:5])
        total = sum(times[:8])
        cpu_times.append((idle, total))
    return cpu_times

  def cpu_usage_percent(self) -> list[float]:
    current_cpu_times = self._read_cpu_times()
    usage = []
    for (last_idle, last_total), (idle, total) in zip(self._last_cpu_times, current_cpu_times, strict=False):
      idle_delta = idle - last_idle
      total_delta = total - last_total
      usage.append(0. if total_delta <= 0 else 100. * (total_delta - idle_delta) / total_delta)
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
    return 100. * (total - memory['MemAvailable:']) / total
