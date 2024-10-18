#pragma once

#include <QNetworkReply>
#include <QString>
#include <QTimer>

#include "common/api.h"

/**
 * Makes a request to the request endpoint.
 */

class HttpRequest : public QObject {
  Q_OBJECT

public:
  enum class Method {GET, DELETE};

  explicit HttpRequest(QObject* parent, bool create_jwt = true, int timeout = 20000);
  void sendRequest(const QString &requestURL, const Method method = Method::GET);
  bool active() const;
  bool timeout() const;

signals:
  void requestDone(const QString &response, bool success, QNetworkReply::NetworkError error);

protected:
  QNetworkReply *reply = nullptr;

private:
  static QNetworkAccessManager *nam();
  QTimer *networkTimer = nullptr;
  bool create_jwt;

private slots:
  void requestTimeout();
  void requestFinished();
};
