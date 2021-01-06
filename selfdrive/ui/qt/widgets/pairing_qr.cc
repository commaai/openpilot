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

#include "pairing_qr.hpp"
#include "common/params.h"
#include "QrCode.hpp"

using qrcodegen::QrCode;

PairingQRWidget::PairingQRWidget(QWidget *parent) : QWidget(parent) {
  qrCode = new QLabel;
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(qrCode);
  this->setLayout(v);
  this->updateQrCode("Hello world!");
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