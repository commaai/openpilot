/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/request_repeater.h"

RequestRepeaterSP::RequestRepeaterSP(QObject *parent, const QString &requestURL, const QString &cacheKey,
                                 int period, bool whileOnroad, bool sunnylink) : HttpRequestSP(parent, true, 20000, sunnylink) {
  request_url = requestURL;
  while_onroad = whileOnroad;
  timer = new QTimer(this);
  timer->setTimerType(Qt::VeryCoarseTimer);
  connect(timer, &QTimer::timeout, [=]() { this->timerTick(); });
  timer->start(period * 1000);

  if (!cacheKey.isEmpty()) {
    prevResp = QString::fromStdString(params.get(cacheKey.toStdString()));
    if (!prevResp.isEmpty()) {
      QTimer::singleShot(500, [=]() { emit requestDone(prevResp, true, QNetworkReply::NoError); });
    }
    connect(this, &HttpRequest::requestDone, [=](const QString &resp, bool success) {
      if (success && resp != prevResp) {
        params.put(cacheKey.toStdString(), resp.toStdString());
        prevResp = resp;
      }
    });
  }

  // Don't wait for the timer to fire to send the first request
  ForceUpdate();
}

void RequestRepeaterSP::timerTick() {
  if ((!uiState()->scene.started || while_onroad) && device()->isAwake() && !active()) {
    LOGD("Sending request for %s", qPrintable(request_url));
    sendRequest(request_url);
  }
}
