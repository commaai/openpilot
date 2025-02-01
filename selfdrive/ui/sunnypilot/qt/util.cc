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
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLayoutItem>
#include <QPainterPath>
#include <QVariant>

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

QMap<QString, QVariantMap> loadPlatformList() {
  QMap<QString, QVariantMap> _platforms;

  std::string json_data = util::read_file("../../sunnypilot/selfdrive/car/car_list.json");

  if (json_data.empty()) {
    return _platforms;
  }

  QJsonParseError json_error{};
  QJsonDocument doc = QJsonDocument::fromJson(QString::fromStdString(json_data).toUtf8(), &json_error);
  if (doc.isNull()) {
    return _platforms;
  }

  if (doc.isObject()) {
    QJsonObject obj = doc.object();
    for (const QString &key : obj.keys()) {
      QJsonObject attributes = obj.value(key).toObject();
      QVariantMap platform_data;

      QJsonArray yearArray = attributes.value("year").toArray();
      QVariantList yearList;
      for (const QJsonValue &year : yearArray) {
        yearList.append(year.toString());
      }

      platform_data["year"] = yearList;
      platform_data["make"] = attributes.value("make").toString();
      platform_data["brand"] = attributes.value("brand").toString();
      platform_data["model"] = attributes.value("model").toString();
      platform_data["platform"] = attributes.value("platform").toString();
      platform_data["package"] = attributes.value("package").toString();

      _platforms[key] = platform_data;
    }
  }

  return _platforms;
}
