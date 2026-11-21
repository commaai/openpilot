#!/usr/bin/env python3
from openpilot.system.ui.lib.application import gui_app
import openpilot.system.ui.tici_reset as tici_reset
import openpilot.system.ui.mici_reset as mici_reset


def main():
  if gui_app.big_ui():
    tici_reset.main()
  else:
    mici_reset.main()


if __name__ == "__main__":
  main()
