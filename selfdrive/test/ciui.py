#!/usr/bin/env python3
import signal
import subprocess

signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from openpilot.selfdrive.ui.qt.python_helpers import set_main_window

class Window(QWidget):
  def __init__(self, parent=None):
    super().__init__(parent)

    layout = QVBoxLayout()
    self.setLayout(layout)

    self.l = QLabel("jenkins runner")
    layout.addWidget(self.l)
    layout.addStretch(1)
    layout.setContentsMargins(20, 20, 20, 20)

    cmds = [
      "cat /etc/hostname",
      "echo AGNOS v$(cat /VERSION)",
      "uptime -p",
    ]
    self.labels = {}
    for c in cmds:
      self.labels[c] = QLabel(c)
      layout.addWidget(self.labels[c])

    self.setStyleSheet("""
      * {
        color: white;
        font-size: 55px;
        background-color: black;
        font-family: "JetBrains Mono";
      }
    """)

    self.timer = QTimer()
    self.timer.timeout.connect(self.update)
    self.timer.start(10 * 1000)
    self.update()

  def update(self):
    for cmd, label in self.labels.items():
      out = subprocess.run(cmd, capture_output=True,
                           shell=True, check=False, encoding='utf8').stdout
      label.setText(out.strip())

if __name__ == "__main__":
  app = QApplication([])
  w = Window()
  set_main_window(w)
  app.exec_()
