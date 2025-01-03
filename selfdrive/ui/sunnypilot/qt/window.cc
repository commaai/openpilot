/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/window.h"

MainWindowSP::MainWindowSP(QWidget *parent)
    : MainWindow(parent, new HomeWindowSP(parent), new SettingsWindowSP(parent)) {

  homeWindow = dynamic_cast<HomeWindowSP *>(MainWindow::homeWindow);
  settingsWindow = dynamic_cast<SettingsWindowSP *>(MainWindow::settingsWindow);
}

void MainWindowSP::closeSettings() {
  MainWindow::closeSettings();
}
