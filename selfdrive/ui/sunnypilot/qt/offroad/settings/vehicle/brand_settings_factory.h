/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/brand_settings_interface.h"

class BrandSettingsFactory {

public:
  static BrandSettingsInterface* createBrandSettings(const QString &brand, QWidget *parent = nullptr);
  static bool isBrandSupported(const QString& brand);
  static QStringList getSupportedBrands();
};
