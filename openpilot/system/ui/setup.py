#!/usr/bin/env python3
from openpilot.system.ui.lib.application import gui_app
import openpilot.system.ui.tici_setup as tici_setup
import openpilot.system.ui.mici_setup as mici_setup


def main():
  if gui_app.big_ui():
    tici_setup.main()
  else:
    mici_setup.main()


if __name__ == "__main__":
  main()
