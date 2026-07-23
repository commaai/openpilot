"""Minimal process set for a passive camera and CAN logger."""

import sys

from openpilot.system.manager.process import ManagerProcess, NativeProcess, PythonProcess


DASHCAM_PROCESS_NAMES = ("logmessaged", "loggerd", "deleter", "pandad", "webcamerad", "encoderd")
USB_PANDAD_MODULE = "openpilot.tools.cm5.usb_pandad"


def always_run(*_args) -> bool:
  return True


def build_dashcam_processes(panda_module: str = USB_PANDAD_MODULE) -> dict[str, ManagerProcess]:
  """Return fresh process objects in safe startup order.

  logmessaged starts first so process diagnostics reach both disk and rlog.
  loggerd starts before CAN and camera publishers. encoderd starts last and
  waits for the road VisionIPC stream.
  """
  processes: list[ManagerProcess] = [
    PythonProcess("logmessaged", "openpilot.system.logmessaged", always_run, separate_process_group=True),
    NativeProcess("loggerd", "openpilot/system/loggerd", ["./loggerd"], always_run, separate_process_group=True),
    PythonProcess("deleter", "openpilot.system.loggerd.deleter", always_run, separate_process_group=True),
    NativeProcess("pandad", ".", [sys.executable, "-m", panda_module], always_run, separate_process_group=True),
    PythonProcess("webcamerad", "openpilot.system.camerad.webcam.camerad", always_run, separate_process_group=True),
    NativeProcess("encoderd", "openpilot/system/loggerd", ["./encoderd"], always_run, separate_process_group=True),
  ]
  return {process.name: process for process in processes}
