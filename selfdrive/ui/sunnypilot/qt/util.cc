/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/util.h"
#include "selfdrive/ui/qt/util.h"

#include <string>
#include <vector>

#include <QDir>
#include <QLayoutItem>
#include <QPainterPath>

#include "system/hardware/hw.h"

std::optional<QString> getParamIgnoringDefault(const std::string &param_name, const std::string &default_value) {
  std::string value = Params().get(param_name);

  if (!value.empty() && value != default_value)
    return QString::fromStdString(value);

  return {};
}

QString getUserAgent(bool sunnylink) {
  return (sunnylink ? "sunnypilot-" : "openpilot-") + getVersion();
}

std::optional<QString> getSunnylinkDongleId() {
  return getParamIgnoringDefault("SunnylinkDongleId", "UnregisteredDevice");
}
