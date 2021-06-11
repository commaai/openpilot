#include "selfdrive/ui/qt/request_repeater.h"

RequestRepeater::RequestRepeater(QObject *parent, const QString &requestURL, const QString &cacheKey,
                                 int period) : HttpRequest(parent, requestURL) {
  timer = new QTimer(this);
  timer->setTimerType(Qt::VeryCoarseTimer);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    if (!QUIState::ui_state.scene.started && QUIState::ui_state.awake && reply == NULL) {
      sendRequest(requestURL);
    }
  });

  timer->start(period * 1000);

  if (!cacheKey.isEmpty()) {
    prevResp = QString::fromStdString(Params().get(cacheKey.toStdString()));
    if (!prevResp.isEmpty()) {
      QTimer::singleShot(0, [=]() { emit receivedResponse(prevResp); });
    }
    QObject::connect(this, &HttpRequest::receivedResponse, [=](const QString &resp) {
      if (resp != prevResp) {
        Params().put(cacheKey.toStdString(), resp.toStdString());
        prevResp = resp;
      }
    });
    QObject::connect(this, &HttpRequest::failedResponse, [=](const QString &err) {
      if (!prevResp.isEmpty()) {
        Params().remove(cacheKey.toStdString());
        prevResp = "";
      }
    });
  }
}
