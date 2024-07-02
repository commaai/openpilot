import ctypes
import os

LINUX = os.name == 'posix' and os.uname().sysname == 'Linux'

if LINUX:
  libc = ctypes.CDLL('libc.so.6')

def setthreadname(name: str) -> None:
  if LINUX:
    name = name[-15:] + '\0'
    libc.prctl(15, str.encode(name), 0, 0, 0)

def getthreadname() -> str:
  if LINUX:
    name = ctypes.create_string_buffer(16)
    libc.prctl(16, name)
    return name.value.decode('utf-8')
  return ""
