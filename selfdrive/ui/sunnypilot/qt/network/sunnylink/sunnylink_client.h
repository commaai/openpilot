/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QObject>

#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/services/role_service.h"
#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/services/user_service.h"

class SunnylinkClient : public QObject {
  Q_OBJECT

public:
  explicit SunnylinkClient(QObject* parent);
  RoleService* role_service;
  UserService* user_service;
};
