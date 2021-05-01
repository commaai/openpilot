#pragma once

#include <QCryptographicHash>
#include <QJsonValue>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPair>
#include <QString>
#include <QVector>
#include <QWidget>

#include <atomic>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

class CommaApi : public QObject {
  Q_OBJECT

public:
  static QByteArray rsa_sign(QByteArray data);
  static QString create_jwt(QVector<QPair<QString, QJsonValue>> payloads = {}, int expiry = 3600);

private:
  QNetworkAccessManager* networkAccessManager;
};

/**
 * Makes a request to the request endpoint.
 */

class HttpRequest : public QObject {
  Q_OBJECT

public:
  explicit HttpRequest(QObject* parent, const QString &requestURL, const QString &cache_key = "", bool create_jwt_ = true);
  QNetworkReply *reply;
  void sendRequest(const QString &requestURL);

private:
  QNetworkAccessManager *networkAccessManager;
  QTimer *networkTimer;
  QString cache_key;
  bool create_jwt;

private slots:
  void requestTimeout();
  void requestFinished();

signals:
  void receivedResponse(const QString &response);
  void failedResponse(const QString &errorString);
  void timeoutResponse(const QString &errorString);
};
