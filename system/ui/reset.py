#!/usr/bin/env python3
from openpilot.system.ui.lib.application import gui_app


def main():
  if gui_app.big_ui():
    import openpilot.system.ui.tici_reset as tici_reset
    tici_reset.main()
  else:
    import openpilot.system.ui.mici_reset as mici_reset
    mici_reset.main()


if __name__ == "__main__":
  main()
