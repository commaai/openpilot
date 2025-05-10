/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/brand_settings_factory.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/brands.h"

static const QStringList supportedBrands = {
  "chrysler",
  "ford",
  "gm",
  "honda",
  "hyundai",
  "mazda",
  "nissan",
  "rivian",
  "subaru",
  "tesla",
  "toyota",
  "volkswagen",
};

BrandSettingsInterface* BrandSettingsFactory::createBrandSettings(const QString& brand, QWidget* parent) {
  if (brand == "chrysler")
    return new ChryslerSettings(parent);
  if (brand == "ford")
    return new FordSettings(parent);
  if (brand == "gm")
    return new GMSettings(parent);
  if (brand == "honda")
    return new HondaSettings(parent);
  if (brand == "hyundai")
    return new HyundaiSettings(parent);
  if (brand == "mazda")
    return new MazdaSettings(parent);
  if (brand == "nissan")
    return new NissanSettings(parent);
  if (brand == "rivian")
    return new RivianSettings(parent);
  if (brand == "subaru")
    return new SubaruSettings(parent);
  if (brand == "tesla")
    return new TeslaSettings(parent);
  if (brand == "toyota")
    return new ToyotaSettings(parent);
  if (brand == "volkswagen")
    return new VolkswagenSettings(parent);

  // Default empty settings if brand not supported
  return nullptr;
}

bool BrandSettingsFactory::isBrandSupported(const QString& brand) {
  return supportedBrands.contains(brand);
}

QStringList BrandSettingsFactory::getSupportedBrands() {
  return supportedBrands;
}
