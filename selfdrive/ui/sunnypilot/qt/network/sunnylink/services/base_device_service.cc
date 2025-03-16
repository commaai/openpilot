/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/network/sunnylink//services/base_device_service.h"

#include "selfdrive/ui/sunnypilot/qt/request_repeater.h"

#include "common/swaglog.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/sunnylink_panel.h"

BaseDeviceService::BaseDeviceService(QObject* parent) : QObject(parent) {
  param_watcher = new ParamWatcher(this);
  connect(param_watcher, &ParamWatcher::paramChanged, [=](const QString &param_name, const QString &param_value) {
    paramsRefresh();
  });
  param_watcher->addParam("SunnylinkEnabled");
}

void BaseDeviceService::paramsRefresh() {
}

void BaseDeviceService::loadDeviceData(const QString &url, bool poll) {
  if (!is_sunnylink_enabled()) {
    LOGW("Sunnylink is not enabled, refusing to load data.");
    return;
  }

  auto sl_dongle_id = getSunnylinkDongleId();
  if (!sl_dongle_id.has_value())
    return;

  QString fullUrl = SUNNYLINK_BASE_URL + "/device/" + *sl_dongle_id + url;
  if (poll && !isCurrentyPolling()) {
    LOGD("Polling %s", qPrintable(fullUrl));
    LOGD("Cache key: SunnylinkCache_%s", qPrintable(QString(getCacheKey())));
    repeater = new RequestRepeaterSP(this, fullUrl, "SunnylinkCache_" + getCacheKey(), 60, false, true);
    connect(repeater, &RequestRepeaterSP::requestDone, this, &BaseDeviceService::handleResponse);
  } else if (isCurrentyPolling()) {
    repeater->ForceUpdate();
  } else {
    LOGD("Sending one-time %s", qPrintable(fullUrl));
    initial_request = new HttpRequestSP(this, true, 10000, true);
    connect(initial_request, &HttpRequestSP::requestDone, this, &BaseDeviceService::handleResponse);
  }
}

void BaseDeviceService::stopPolling() {
  if (repeater != nullptr) {
    repeater->deleteLater();
    repeater = nullptr;
  }
}
