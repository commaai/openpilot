/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QWidget>

class BrandSettingsInterface : public QWidget {
  Q_OBJECT

public:
  explicit BrandSettingsInterface(QWidget *parent = nullptr) : QWidget(parent) {}
  virtual ~BrandSettingsInterface() = default;

  virtual void updatePanel(bool offroad) = 0;
  virtual void updateSettings() = 0;
};
