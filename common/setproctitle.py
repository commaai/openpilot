import os
from openpilot.common.run import run_cmd

def getproctitle():
    pid = os.getpid()
    proctitle = run_cmd([f"cat /proc/{pid}/comm"])
    return proctitle

def setproctitle(title: str):
    pid = os.getpid()
    run_cmd([f"echo {title} > /proc/{pid}/comm"])
