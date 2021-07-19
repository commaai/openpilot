#pragma once

#include <QJsonObject>
#include <QNetworkReply>
#include <QString>
#include <QTimer>

namespace CommaApi {

QByteArray rsa_sign(const QByteArray &data);
QString create_jwt(const QJsonObject &payloads = {}, int expiry = 3600);

}  // namespace CommaApi

/**
 * Makes a request to the request endpoint.
 */

class HttpRequest : public QObject {
  Q_OBJECT

public:
  explicit HttpRequest(QObject* parent, const QString &requestURL, bool create_jwt_ = true);
  void sendRequest(const QString &requestURL);

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *networkAccessManager;
  QTimer *networkTimer;
  bool create_jwt;

private slots:
  void requestTimeout();
  void requestFinished();

signals:
  void receivedResponse(const QString &response);
  void failedResponse(const QString &errorString);
  void timeoutResponse(const QString &errorString);
};
