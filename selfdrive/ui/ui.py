#!/usr/bin/env python3
import os
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

import cereal.messaging as messaging
from openpilot.system.hardware import HARDWARE

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QStackedLayout, QApplication
from openpilot.selfdrive.ui.qt.python_helpers import set_main_window


if __name__ == "__main__":
  app = QApplication([])
  win = QWidget()
  set_main_window(win)

  bg = QLabel("", alignment=Qt.AlignCenter)

  alert1 = QLabel()
  alert2 = QLabel()
  vlayout = QVBoxLayout()
  vlayout.addWidget(alert1, alignment=Qt.AlignCenter)
  vlayout.addWidget(alert2, alignment=Qt.AlignCenter)

  tmp = QWidget()
  tmp.setLayout(vlayout)

  stack = QStackedLayout(win)
  stack.addWidget(tmp)
  stack.addWidget(bg)
  stack.setStackingMode(QStackedLayout.StackAll)

  win.setObjectName("win")
  win.setStyleSheet("""
    #win {
      background-color: black;
    }
    QLabel {
      color: white;
      font-size: 40px;
    }
  """)

  sm = messaging.SubMaster(['deviceState', 'controlsState'])

  def update():
    sm.update(0)

    onroad = sm.all_checks(['deviceState']) and sm['deviceState'].started
    if onroad:
      cs = sm['controlsState']
      color = ("grey" if str(cs.state) in ("overriding", "preEnabled") else "green") if cs.enabled else "blue"
      bg.setText("\U0001F44D" if cs.engageable else "\U0001F6D1")
      bg.setStyleSheet(f"font-size: 100px; background-color: {color};")
      bg.show()

      alert1.setText(cs.alertText1)
      alert2.setText(cs.alertText2)

      if not sm.alive['controlsState']:
        alert1.setText("waiting for controls...")
    else:
      bg.hide()
      alert1.setText("")
      alert2.setText("offroad")

    HARDWARE.set_screen_brightness(100 if onroad else 40)
    os.system("echo 0 > /sys/class/backlight/panel0-backlight/bl_power")

  timer = QTimer()
  timer.timeout.connect(update)
  timer.start(50)

  app.exec_()
