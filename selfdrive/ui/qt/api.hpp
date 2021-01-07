#pragma once

#include <QWidget>
#include <QVector>
#include <QPair>
#include <QString>
#include <QJsonValue>
#include <QCryptographicHash>

#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>


class CommaApi{
public:
  static QByteArray rsa_sign(QByteArray data);
  static QString create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry = 3600);
};
