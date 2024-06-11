import ctypes
import os

LINUX = os.name == 'posix' and os.uname().sysname == 'Linux'

def setthreadname(name: str) -> None:
  if LINUX:
    libc = ctypes.CDLL('libc.so.6')
    libc.prctl(15, str.encode(name), 0, 0, 0)

def getthreadname() -> str:
  if LINUX:
    with open('/proc/self/comm') as f:
      return f.read().strip()
  return ""
