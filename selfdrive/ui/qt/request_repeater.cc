#include "request_repeater.hpp"

RequestRepeater::RequestRepeater(QObject *parent, QString requestURL, const QString &cache_key, int period_seconds, bool disableWithScreen) :
  HttpRequest(parent, requestURL, cache_key), disableWithScreen(disableWithScreen) {
  QTimer* timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, [=](){
    if (!uiThread()->onroad() && reply == NULL && (uiThread()->awake() || !disableWithScreen)) {
      sendRequest(requestURL);
    }
  });
  timer->start(period_seconds * 1000);
}
