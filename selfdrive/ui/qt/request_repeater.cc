#include "request_repeater.hpp"

RequestRepeater::RequestRepeater(QObject *parent, QString requestURL, const QString &cacheKey,
                                 int period) : HttpRequest(parent, requestURL, cacheKey) {
  timer = new QTimer(this);
  timer->setTimerType(Qt::VeryCoarseTimer);
  QObject::connect(timer, &QTimer::timeout, [=](){
    // TODO: add back screen awake check
    if (!QUIState::ui_state.scene.started && reply == NULL) {
      sendRequest(requestURL);
    }
  });
  timer->start(period * 1000);
}
