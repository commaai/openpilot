#pragma once

#include <QJsonObject>
#include <QNetworkReply>
#include <QString>
#include <QTimer>

#include "util.h"
#include "common/util.h"

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
  enum class Method {GET, DELETE, POST, PUT};

  explicit HttpRequest(QObject* parent, bool create_jwt = true, int timeout = 20000);
  virtual void sendRequest(const QString &requestURL, Method method);
  void sendRequest(const QString &requestURL) { sendRequest(requestURL, Method::GET);}
  bool active() const;
  bool timeout() const;

signals:
  void requestDone(const QString &response, bool success, QNetworkReply::NetworkError error);

protected:
  QNetworkReply *reply = nullptr;
  static QNetworkAccessManager *nam();
  QTimer *networkTimer = nullptr;
  bool create_jwt;
  virtual QNetworkRequest prepareRequest(const QString& requestURL);
  [[nodiscard]] virtual QString GetJwtToken() const { return CommaApi::create_jwt(); }
  [[nodiscard]] virtual QString GetUserAgent() const { return getUserAgent(); }

protected slots:
  void requestTimeout();
  void requestFinished();
};
