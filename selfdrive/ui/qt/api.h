#pragma once

#include <QJsonObject>
#include <QNetworkReply>
#include <QString>
#include <QTimer>

#include "selfdrive/common/util.h"

namespace CommaApi {

const QString BASE_URL = util::getenv("API_HOST", "https://api.commadotai.com").c_str();
QByteArray rsa_sign(const QByteArray &data);
QString create_jwt(const QJsonObject &payloads = {}, int expiry = 3600);

}  // namespace CommaApi

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
  void requestDone(const QString &response, bool success);

protected:
  QNetworkReply *reply = nullptr;

private:
  QNetworkAccessManager *networkAccessManager = nullptr;
  QTimer *networkTimer = nullptr;
  bool create_jwt;

private slots:
  void requestTimeout();
  void requestFinished();
};
