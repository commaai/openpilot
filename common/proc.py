#!/usr/bin/env python3
import ctypes
import ctypes.util

def setproctitle(title: str) -> None:
  try:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    buff = ctypes.create_string_buffer(len(title)+1)
    buff.value = title.encode()
    libc.prctl(15, ctypes.byref(buff), 0, 0, 0)
  except OSError:
    pass
