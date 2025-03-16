/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/request_repeater.h"

#include "common/swaglog.h"
#include "common/util.h"
#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/api.h"

class RequestRepeaterSP : public HttpRequestSP {

private:
  Params params;
  QTimer *timer;
  QString prevResp;
  QString request_url;
  bool while_onroad;
  void timerTick();

public:
  RequestRepeaterSP(QObject *parent, const QString &requestURL, const QString &cacheKey = "", int period = 0, bool whileOnroad=false, bool sunnylink = false);
  void ForceUpdate() {
    LOGD("Forcing update for %s", qPrintable(request_url));
    timerTick();
  }
};
