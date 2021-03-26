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
class TimeoutRequest : public QObject {
  Q_OBJECT

public:
  explicit TimeoutRequest(QWidget* parent, const QString &requestURL, const QString &cache_key = "", int timeout_second = 20, QVector<QPair<QString, QJsonValue>> payloads = *(new QVector<QPair<QString, QJsonValue>>()), bool disableWithScreen = true);
  void setRepeatPeriod(int repeat_period_second);
  bool active = true;

private:
  bool disableWithScreen;
  QNetworkReply* reply = nullptr;
  QNetworkAccessManager* networkAccessManager = nullptr;
  QTimer* networkTimer = nullptr;
  QTimer* repeatTimer = nullptr;
  QString cache_key;
  void sendRequest(QString requestURL);

private slots:
  void requestTimeout();
  void requestFinished();

signals:
  void receivedResponse(const QString &response);
  void failedResponse(const QString &errorString);
  void timeoutResponse(const QString &errorString);
};
