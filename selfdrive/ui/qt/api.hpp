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
  static QString create_jwt(QVector<QPair<QString, QJsonValue>> payloads = {}, int expiry=3600);

private:
  QNetworkAccessManager* networkAccessManager;
};

/**
 * Makes repeated requests to the request endpoint.
 */
class RequestRepeater : public QObject {
  Q_OBJECT

public:
  explicit RequestRepeater(QWidget* parent, QString requestURL, int period = 10, const QString &cache_key = "", bool disableWithScreen = true);
  bool active = true;

private:
  bool disableWithScreen;
  QNetworkReply* reply;
  QNetworkAccessManager* networkAccessManager;
  QTimer* networkTimer;
  QString cache_key;
  void sendRequest(QString requestURL);

private slots:
  void requestTimeout();
  void requestFinished();

signals:
  void receivedResponse(QString response);
  void failedResponse(QString errorString);
  void timeoutResponse(QString errorString);
};
