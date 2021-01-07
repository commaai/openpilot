#include <cassert>
#include <iostream>

#include <QDebug>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QVBoxLayout>
#include <QLabel>
#include <QCryptographicHash>
#include <QFile>
#include <QDateTime>

#include "pairing_qr.hpp"
#include "common/params.h"
#include "common/utilpp.h"
#include "QrCode.hpp"
#include "api.hpp"

using qrcodegen::QrCode;

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif


PairingQRWidget::PairingQRWidget(QWidget *parent) : QWidget(parent) {
  qrCode = new QLabel;
  qrCode->setFixedSize(700, 700);
  qrCode->setScaledContents(true);
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(qrCode);
  this->setLayout(v);

  QString IMEI = QString::fromStdString(Params().get("IMEI"));
  QString serial = QString::fromStdString(Params().get("HardwareSerial"));

  QVector<QPair<QString, QJsonValue>> payloads;
  payloads.push_back(qMakePair(QString("pair"), true));
  QString pairToken = CommaApi::create_jwt(payloads);
  
  QString qrString = IMEI + "--" + serial + "--" + pairToken;
  qDebug() << qrString;
  this->updateQrCode(qrString);
}

void PairingQRWidget::updateQrCode(QString text) {
  QrCode qr = QrCode::encodeText( text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  // We make the image larger so we can have a white border
  QImage im(sz+2,sz+2, QImage::Format_RGB32);
  QRgb black = qRgb(  0,  0,  0);
  QRgb white = qRgb(255,255,255);

  for (int y = 0; y < sz+2; y++) {
    for (int x = 0; x < sz+2; x++) {
      im.setPixel(x, y, white);
    }
  }
  for (int y = 0; y < sz; y++) {
    for (int x = 0; x < sz; x++) {
      im.setPixel(x+1,y+1,qr.getModule(x, y) ? black : white );
    }
  }
  qrCode->setPixmap( QPixmap::fromImage(im.scaled(256,256,Qt::KeepAspectRatio,Qt::FastTransformation),Qt::MonoOnly) );
}
