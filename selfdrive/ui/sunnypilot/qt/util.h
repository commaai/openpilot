/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <optional>
#include <vector>

#include <QMap>
#include <QPainter>
#include <QWidget>

QString getUserAgent(bool sunnylink = false);
std::optional<QString> getSunnylinkDongleId();
std::optional<QString> getParamIgnoringDefault(const std::string &param_name, const std::string &default_value);
QMap<QString, QVariantMap> loadPlatformList();
