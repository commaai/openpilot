#pragma once

#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/ui.h"

class RequestRepeater : public QObject {
  Q_OBJECT

public:
  HttpRequest *request(const QString &requestURL, const QString &cacheKey = "", int period = 0, bool while_onroad = false);

public slots:
  void offroadTransition(bool offroad);
  void displayPowerChanged(bool on);

private:
  RequestRepeater() : QObject() {};
  void updateRequests();

  struct Request {
    bool while_onroad;
    QString url;
    QTimer timer;
    HttpRequest req;
    QString prevResp;
  };
  std::vector<std::unique_ptr<Request>> requests_;
  Params params_;
  bool offroad_ = true, awake_ = true;

  friend RequestRepeater *requestRepeater();
};

RequestRepeater *requestRepeater();
