/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/max_time_offroad.h"

// Map of Max Offroad Time Options (Minutes)
const QMap<QString, QString> MaxTimeOffroad::offroad_time_options = {
  {"0", "0"}, // Always On
  {"1", "5"},
  {"2", "10"},
  {"3", "15"},
  {"4", "30"},
  {"5", "60"},
  {"6", "120"},
  {"7", "180"},
  {"8", "300"},
  {"9", "600"},
  {"10", "1440"},
  {"11", "1800"}
};

MaxTimeOffroad::MaxTimeOffroad() : OptionControlSP(
  "MaxTimeOffroad",
  tr("Max Time Offroad"),
  tr("Device will automatically shutdown after set time once the engine is turned off.<br/>(30h is the default)"),
  "../assets/offroad/icon_blank.png",
  {0, 11}, 1, true, &offroad_time_options) {

  refresh();
}

void MaxTimeOffroad::refresh() {
  const int maxOffroadInMinutes = QString::fromStdString(params.get("MaxTimeOffroad")).toInt();
  const bool useHours = maxOffroadInMinutes >= 60;
  
  QString label;
  if (maxOffroadInMinutes == 0) {
    label = tr("Always On");
  } else {
    const int value = useHours ? maxOffroadInMinutes / 60 : maxOffroadInMinutes;
    label = QString("%1%2").arg(value).arg(useHours ? tr("h") : tr("m"));
  }
  
  if (maxOffroadInMinutes == 1800) {
    label += tr(" (default)");
  }
  
  setLabel(label);
}
