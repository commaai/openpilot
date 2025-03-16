/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QJsonObject>

class UserModel {
public:
  QString device_id;
  QString user_id;
  qint64 created_at;
  qint64 updated_at;
  QString token_hash;

  explicit UserModel(const QJsonObject &json) {
    device_id = json["device_id"].toString();
    user_id = json["user_id"].toString();
    created_at = json["created_at"].toInt();
    updated_at = json["updated_at"].toInt();
    token_hash = json["token_hash"].toString();
  }

  [[nodiscard]] QJsonObject toJson() const {
    QJsonObject json;
    json["device_id"] = device_id;
    json["user_id"] = user_id;
    json["created_at"] = created_at;
    json["updated_at"] = updated_at;
    json["token_hash"] = token_hash;
    return json;
  }
};
