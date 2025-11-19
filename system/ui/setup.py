#!/usr/bin/env python3
from openpilot.system.ui.lib.application import gui_app


def main():
  if gui_app.big_ui():
    import openpilot.system.ui.tici_setup as tici_setup
    tici_setup.main()
  else:
    import openpilot.system.ui.mici_setup as mici_setup
    mici_setup.main()


if __name__ == "__main__":
  main()
