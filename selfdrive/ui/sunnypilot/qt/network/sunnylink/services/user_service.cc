/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/services/user_service.h"

#include <QJsonArray>
#include <QJsonDocument>

#include "selfdrive/ui/sunnypilot/ui.h"

UserService::UserService(QObject* parent) : BaseDeviceService(parent) {
  url = "/users";
}

void UserService::load() {
  loadDeviceData(url);
}

void UserService::startPolling() {
  loadDeviceData(url, true);
}

void UserService::handleResponse(const QString &response, bool success) {
  if (!success) {
    return;
  }

  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  QJsonArray jsonArray = doc.array();

  std::vector<UserModel> users;
  for (const auto &value : jsonArray) {
    users.emplace_back(value.toObject());
  }

  emit usersReady(users);
  uiStateSP()->setSunnylinkDeviceUsers(users);
}
