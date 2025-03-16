/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"

class BaseDeviceService : public QObject {
  Q_OBJECT

protected:
  void paramsRefresh();
  void loadDeviceData(const QString &url, bool poll = false);
  virtual void handleResponse(const QString &response, bool success) = 0;

  static bool is_sunnylink_enabled() { return Params().getBool("SunnylinkEnabled");}
  ParamWatcher* param_watcher;
  HttpRequestSP* initial_request = nullptr;
  RequestRepeaterSP* repeater = nullptr;

public:
  explicit BaseDeviceService(QObject* parent = nullptr);
  virtual QString getCacheKey() const = 0;
  bool isCurrentyPolling() {return repeater != nullptr;}
  void stopPolling();
};
