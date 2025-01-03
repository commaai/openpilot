#pragma once

#include "common/util.h"
#include "selfdrive/ui/qt/api.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/ui.h"
#else
#include "selfdrive/ui/ui.h"
#endif

class RequestRepeater : public HttpRequest {
public:
  RequestRepeater(QObject *parent, const QString &requestURL, const QString &cacheKey = "", int period = 0, bool while_onroad=false);

private:
  Params params;
  QTimer *timer;
  QString prevResp;
};
