/**
* Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <map>
#include <string>

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"

class SunnypilotPanel : public ListWidget {
  Q_OBJECT

public:
  explicit SunnypilotPanel(SettingsWindowSP *parent = nullptr);

private:
  QStackedLayout* main_layout = nullptr;
};
