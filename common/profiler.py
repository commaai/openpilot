import time

class Profiler():
  def __init__(self, enabled=False):
    self.enabled = enabled
    self.cp = {}
    self.cp_ignored = []
    self.iter = 0
    self.start_time = time.time()
    self.last_time = self.start_time
    self.tot = 0.

  def reset(self, enabled=False):
    self.enabled = enabled
    self.cp = {}
    self.cp_ignored = []
    self.iter = 0
    self.start_time = time.time()
    self.last_time = self.start_time

  def checkpoint(self, name, ignore=False):
    # ignore flag needed when benchmarking threads with ratekeeper
    if not self.enabled:
      return
    tt = time.time()
    if name not in self.cp:
      self.cp[name] = 0.
      if ignore:
        self.cp_ignored.append(name)
    self.cp[name] += tt - self.last_time
    if not ignore:
      self.tot += tt - self.last_time
    self.last_time = tt

  def display(self):
    if not self.enabled:
      return
    self.iter += 1
    print("******* Profiling %d *******" % self.iter)
    for n, ms in sorted(self.cp.items(), key=lambda x: -x[1]):
      if n in self.cp_ignored:
        print("%30s: %9.2f  avg: %7.2f  percent: %3.0f   IGNORED" % (n, ms*1000.0, ms*1000.0/self.iter, ms/self.tot*100))
      else:
        print("%30s: %9.2f  avg: %7.2f  percent: %3.0f" % (n, ms*1000.0, ms*1000.0/self.iter, ms/self.tot*100))
    print(f"Iter clock: {self.tot / self.iter:2.6f}   TOTAL: {self.tot:2.2f}")
