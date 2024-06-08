import ctypes

libc = ctypes.CDLL('libc.so.6')

PRCTL_CALL = 15

def setproctitle(name: str):
  libc.prctl(PRCTL_CALL, str.encode(name), 0, 0, 0)

def getproctitle():
  with open('/proc/self/comm') as f:
    process_name = f.read().strip()
  return process_name
