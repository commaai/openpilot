#pragma once

#include <QWidget>
#include <QCryptographicHash>

#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>

class CommaApi{
public:
  static QByteArray rsa_sign(QByteArray data); 
};
