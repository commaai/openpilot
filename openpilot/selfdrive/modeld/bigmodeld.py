#!/usr/bin/env python3
import subprocess

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld import modeld


def main() -> None:
  try:
    subprocess.run(["udevadm", "settle"], check=True)
    modeld.main(model_name="bigModelV2")
  except Exception:
    cloudlog.exception("big model failed")
    Params().put_bool("UsbGpuFailed", True, block=True)


if __name__ == "__main__":
  main()
