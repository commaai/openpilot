#pragma once

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/ui.h"

class RequestRepeater : public HttpRequest {
public:
  RequestRepeater(QObject *parent, const QString &requestURL, const QString &cacheKey = "", int period = 0);

private:
  QTimer *timer;
};
