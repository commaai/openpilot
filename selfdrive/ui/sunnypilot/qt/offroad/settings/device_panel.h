/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class DevicePanelSP : public DevicePanel {
  Q_OBJECT

public:
  explicit DevicePanelSP(SettingsWindowSP *parent = 0);
  void showEvent(QShowEvent *event) override;
  void setOffroadMode();
  void updateState();
  void resetSettings();

private:
  std::map<QString, PushButtonSP*> buttons;
  PushButtonSP *offroadBtn;

  const QString alwaysOffroadStyle = R"(
    PushButtonSP {
      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;
      color: #FFFFFF;
      background-color: #393939;
    }
    PushButtonSP:pressed {
      background-color: #4A4A4A;
    }
  )";

  const QString autoOffroadStyle = R"(
    PushButtonSP {
      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;
      color: #FFFFFF;
      background-color: #E22C2C;
    }
    PushButtonSP:pressed {
      background-color: #FF2424;
    }
  )";

  const QString rebootButtonStyle = R"(
    PushButtonSP {
      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;
      color: #FFFFFF;
      background-color: #393939;
    }
    PushButtonSP:pressed {
      background-color: #4A4A4A;
    }
  )";

  const QString powerOffButtonStyle = R"(
    PushButtonSP {
      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;
      color: #FFFFFF;
      background-color: #E22C2C;
    }
    PushButtonSP:pressed {
      background-color: #FF2424;
    }
  )";
};
