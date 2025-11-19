#!/usr/bin/env python3
from openpilot.system.ui.lib.application import gui_app


def main():
  if gui_app.big_ui():
    import openpilot.system.ui.tici_updater as tici_updater
    tici_updater.main()
  else:
    import openpilot.system.ui.mici_updater as mici_updater
    mici_updater.main()


if __name__ == "__main__":
  main()
