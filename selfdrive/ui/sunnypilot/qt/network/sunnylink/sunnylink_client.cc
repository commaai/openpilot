/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/sunnylink_client.h"
#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/services/user_service.h"

SunnylinkClient::SunnylinkClient(QObject* parent) : QObject(parent) {
  role_service = new RoleService(parent);
  user_service = new UserService(parent);
}
