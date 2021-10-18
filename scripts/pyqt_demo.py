#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QLabel  # pylint: disable=no-name-in-module, import-error
from selfdrive.ui.qt.python_helpers import set_main_window


if __name__ == "__main__":
  app = QApplication([])
  label = QLabel('Hello World!')

  # Set full screen and rotate
  set_main_window(label)

  app.exec_()
