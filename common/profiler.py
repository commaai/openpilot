from common.realtime import sec_since_boot

class Profiler(object):
  def __init__(self, enabled=False):
    self.enabled = enabled
    self.cp = []
    self.start_time = sec_since_boot()
    self.last_time = self.start_time

  def checkpoint(self, name):
    if not self.enabled:
      return
    tt = sec_since_boot()
    self.cp.append((name, tt - self.last_time))
    self.last_time = tt

  def display(self):
    if not self.enabled:
      return
    print "******* Profiling *******"
    tot = 0.0
    for n, ms in self.cp:
      print "%30s: %7.2f" % (n, ms*1000.0)
      tot += ms
    print "    TOTAL: %7.2f" % (tot*1000.0)

