#include "selfdrive/ui/qt/request_repeater.h"

#include <QApplication>

RequestRepeater::RequestRepeater(QObject *parent) : QObject(parent) {}

HttpRequest *RequestRepeater::request(const QString &url, const QString &cacheKey, int period, bool while_onroad) {
  Request *r = new Request{.url = url,
                           .req = new HttpRequest(this),
                           .timer = new QTimer(this),
                           .while_onroad = while_onroad};
  requests_.emplace_back(r);

  if (!cacheKey.isEmpty()) {
    r->prevResp = QString::fromStdString(params_.get(cacheKey.toStdString()));
    if (!r->prevResp.isEmpty()) {
      QTimer::singleShot(500, [=]() { emit r->req->receivedResponse(r->prevResp); });
    }

    QObject::connect(r->req, &HttpRequest::receivedResponse, [=](const QString &resp) {
      if (resp != r->prevResp) {
        params_.put(cacheKey.toStdString(), resp.toStdString());
        r->prevResp = resp;
      }
    });
  }

  r->timer->setTimerType(Qt::VeryCoarseTimer);
  r->timer->setInterval(period * 1000);
  r->timer->callOnTimeout([=]() {
    if (!r->req->active()) {
      r->req->sendRequest(r->url);
    }
  });
  r->timer->start();
  return r->req;
}

void RequestRepeater::offroadTransition(bool offroad) {
  if (offroad_ != offroad) {
    offroad_ = offroad;
    updateRequests();
  }
}

void RequestRepeater::displayPowerChanged(bool on) {
  if (awake_ != on) {
    awake_ = on;
    updateRequests();
  }
}

void RequestRepeater::updateRequests() {
  for (auto &r : requests_) {
    if (awake_ && (offroad_ || r->while_onroad)) {
      r->timer->start();
     } else {
      r->timer->stop();
     }
  }
}

RequestRepeater *requestRepeater() {
  static RequestRepeater repeater(qApp);
  return &repeater;
}
