#include "selfdrive/ui/qt/request_repeater.h"

RequestRepeater::RequestRepeater(QObject *parent, const QString &requestURL, const QString &cacheKey,
                                 int period, bool while_onroad) : HttpRequest(parent, requestURL) {
  timer = new QTimer(this);
  timer->setTimerType(Qt::VeryCoarseTimer);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    if ((!QUIState::ui_state.scene.started || while_onroad) && QUIState::ui_state.awake && reply == NULL) {
      sendRequest(requestURL);
    }
  });

  timer->start(period * 1000);

  if (!cacheKey.isEmpty()) {
    prevResp = QString::fromStdString(params.get(cacheKey.toStdString()));
    if (!prevResp.isEmpty()) {
      QTimer::singleShot(0, [=]() { emit receivedResponse(prevResp); });
    }
    QObject::connect(this, &HttpRequest::receivedResponse, [=](const QString &resp) {
      if (resp != prevResp) {
        params.put(cacheKey.toStdString(), resp.toStdString());
        prevResp = resp;
      }
    });
    QObject::connect(this, &HttpRequest::failedResponse, [=](const QString &err) {
      if (!prevResp.isEmpty()) {
        params.remove(cacheKey.toStdString());
        prevResp = "";
      }
    });
  }
}
