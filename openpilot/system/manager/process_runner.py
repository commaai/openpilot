import importlib
import os
import sys

from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata
import openpilot.system.sentry as sentry


def main() -> None:
  module, name = sys.argv[1:]
  build_metadata = get_build_metadata()

  sentry.init(sentry.SentryProject.SELFDRIVE)
  sentry.set_tag("daemon", name)
  cloudlog.bind_global(dongle_id=os.getenv("DONGLE_ID"),
                       version=build_metadata.openpilot.version,
                       origin=build_metadata.openpilot.git_normalized_origin,
                       branch=build_metadata.channel,
                       commit=build_metadata.openpilot.git_commit,
                       dirty=build_metadata.openpilot.is_dirty,
                       device=HARDWARE.get_device_type())
  cloudlog.bind(daemon=name)

  try:
    importlib.import_module(module).main()
  except KeyboardInterrupt:
    cloudlog.warning(f"child {module} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise


if __name__ == "__main__":
  main()
