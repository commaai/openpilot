/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <map>
#include <string>

#include "selfdrive/ui/qt/offroad/settings.h"

class SettingsWindowSP : public SettingsWindow {
  Q_OBJECT

public:
  explicit SettingsWindowSP(QWidget *parent = nullptr);

protected:
  struct PanelInfo {
    QString name;
    QWidget *widget;
    QString icon;

    PanelInfo(const QString &name, QWidget *widget, const QString &icon) : name(name), widget(widget), icon(icon) {}
  };
};

class TogglesPanelSP : public TogglesPanel {
  Q_OBJECT

public:
  explicit TogglesPanelSP(SettingsWindowSP *parent);

private slots:
  void updateState(const UIStateSP &s);
};
