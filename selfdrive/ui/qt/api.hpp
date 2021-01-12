#pragma once

#include <QCryptographicHash>
#include <QJsonValue>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPair>
#include <QString>
#include <QVector>
#include <QWidget>

#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

class CommaApi : public QObject {
  Q_OBJECT
public:
  explicit CommaApi(QWidget* parent);
  QByteArray rsa_sign(QByteArray data);
  QString create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry = 3600);
  QString create_jwt();
  QNetworkReply* get(QNetworkRequest request);

private:
  QNetworkAccessManager* networkAccessManager;
};

/**
 * Makes repeated requests to the request endpoint. 
 */
class RequestRepeater : public QObject {
  Q_OBJECT
public:
  explicit RequestRepeater(QWidget* parent, QString requestURL, int period = 10, QVector<QPair<QString, QJsonValue>> payloads = *(new QVector<QPair<QString, QJsonValue>>()));
  bool disableWithScreen = true;
  bool active = true;

private:
  QNetworkAccessManager* networkAccessManager;
  QNetworkReply* reply = NULL;
  CommaApi* api;
  void sendRequest(QString requestURL, QVector<QPair<QString, QJsonValue>> payloads);

private slots:
  void replyFinished();

signals:
  void receivedResponse(QString response);
};