#include "selfdrive/ui/qt/request_repeater.h"

RequestRepeater::RequestRepeater(QObject *parent, const QString &requestURL, const QString &cacheKey,
                                 int period, bool while_onroad) : HttpRequest(parent) {
  timer = new QTimer(this);
  timer->setTimerType(Qt::VeryCoarseTimer);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    if ((!uiState()->scene.started || while_onroad) && device()->isAwake() && !active()) {
      sendRequest(requestURL);
    }
  });

  timer->start(period * 1000);

  if (!cacheKey.isEmpty()) {
    prevResp = QString::fromStdString(params.get(cacheKey.toStdString()));
    if (!prevResp.isEmpty()) {
      QTimer::singleShot(500, [=]() { emit requestDone(prevResp, true, QNetworkReply::NoError); });
    }
    QObject::connect(this, &HttpRequest::requestDone, [=](const QString &resp, bool success) {
      if (success && resp != prevResp) {
        params.put(cacheKey.toStdString(), resp.toStdString());
        prevResp = resp;
      }
    });
  }
}
