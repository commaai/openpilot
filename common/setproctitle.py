import os
from sys import platform
from openpilot.common.run import run_cmd

def getproctitle():
  if platform.startswith("linux"):
    pid = os.getpid()
    proctitle = run_cmd(["cat", f"/proc/{pid}/comm"])
    return proctitle

def setproctitle(title: str):
  if platform.startswith("linux"):
    pid = os.getpid()

    proc_comm = open(f"/proc/{pid}/comm", "w")
    proc_comm.write(title)
    proc_comm.close()
