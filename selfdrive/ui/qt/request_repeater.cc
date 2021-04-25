#include "request_repeater.hpp"

RequestRepeater::RequestRepeater(QObject *parent, QString requestURL, const QString &cache_key, int period_seconds, bool disableWithScreen) :
  HttpRequest(parent, requestURL, cache_key), disableWithScreen(disableWithScreen) {
  QTimer* timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, [=](){
    if (!GLWindow::ui_state.scene.started && reply == NULL && (GLWindow::ui_state.awake || !disableWithScreen)) {
      sendRequest(requestURL);
    }
  });
  timer->start(period_seconds * 1000);
}
