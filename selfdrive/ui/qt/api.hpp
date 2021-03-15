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
  static QString create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry=3600);
  static QString create_jwt();

private:
  QNetworkAccessManager* networkAccessManager;
};

/**
 * Makes repeated requests to the request endpoint.
 */
class RequestRepeater : public QObject {
  Q_OBJECT

public:
  explicit RequestRepeater(QWidget* parent, QString requestURL, int period = 10, QVector<QPair<QString, QJsonValue>> payloads = *(new QVector<QPair<QString, QJsonValue>>()), bool disableWithScreen = true);
  bool active = true;

private:
  bool disableWithScreen;
  QNetworkReply* reply;
  QNetworkAccessManager* networkAccessManager;
  QTimer* networkTimer;
  std::atomic<bool> aborted = false; // Not 100% sure we need atomic
  void sendRequest(QString requestURL, QVector<QPair<QString, QJsonValue>> payloads);

private slots:
  void requestTimeout();
  void requestFinished();

signals:
  void receivedResponse(QString response);
  void failedResponse(QString errorString);
};
