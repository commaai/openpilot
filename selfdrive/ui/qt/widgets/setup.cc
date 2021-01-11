#include <QDebug>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QLabel>
#include <QTimer>

#include "setup.hpp"
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
  CommaApi *api = new CommaApi(this);
  qrCode = new QLabel;
  qrCode->setScaledContents(true);
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(qrCode);
  setLayout(v);

  QString IMEI = QString::fromStdString(Params().get("IMEI"));
  QString serial = QString::fromStdString(Params().get("HardwareSerial"));

  QVector<QPair<QString, QJsonValue>> payloads;
  payloads.push_back(qMakePair(QString("pair"), true));
  QString pairToken = api->create_jwt(payloads);
  
  QString qrString = IMEI + "--" + serial + "--" + pairToken;
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

  int approx500 = (500/(sz+2))*(sz+2);
  qrCode->setPixmap( QPixmap::fromImage(im.scaled(approx500, approx500 ,Qt::KeepAspectRatio, Qt::FastTransformation), Qt::MonoOnly) );
  qrCode->setFixedSize(approx500, approx500);
}


PrimeUserWidget::PrimeUserWidget(QWidget *parent) : QWidget(parent){
  api = new CommaApi(this);

  mainLayout = new QVBoxLayout;
  QLabel *commaPrime = new QLabel("COMMA PRIME");
  commaPrime->setStyleSheet(R"(
    font-size: 60px;
  )");
  mainLayout->addWidget(commaPrime);

  username = new QLabel("Waiting for username");
  mainLayout->addWidget(username);

  mainLayout->addSpacing(200);
  
  QLabel *commaPoints = new QLabel("COMMA POINTS");
  commaPoints->setStyleSheet(R"(
    font-size: 60px;
    color: #b8b8b8;
  )");
  mainLayout->addWidget(commaPoints);

  points = new QLabel("Waiting for comma points");
  mainLayout->addWidget(points);

  setLayout(mainLayout);
  QTimer *timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(1000);
  refresh();
}

void PrimeUserWidget::refresh(){
  QString token = api->create_jwt();

  QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1/devices/" + dongle_id + "/owner"));
  request.setRawHeader("Authorization", ("JWT "+token).toUtf8());
  if(reply == NULL){
    reply = api->get(request);
    connect(reply, &QNetworkReply::finished, this, &PrimeUserWidget::replyFinished);
  }else{
    qDebug()<<"Too many requests, previous request was not yet removed";
  }
}

void PrimeUserWidget::replyFinished() {
  QString answer = reply->readAll();
  
  QJsonDocument doc = QJsonDocument::fromJson(answer.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting username and points";
    reply->deleteLater();
    reply = NULL;
    return;
  }
  QJsonObject json = doc.object();
  QString username_str = json["username"].toString();
  if (username_str.length()) {
    username_str = "@" + username_str;
  }
  QString points_str = QString::number(json["points"].toInt());

  username->setText(username_str);
  points->setText(points_str);
  reply->deleteLater();
  reply = NULL;
}

PrimeAdWidget::PrimeAdWidget(QWidget *parent) : QWidget(parent){
  QVBoxLayout *vlayout = new QVBoxLayout();

  QLabel *upgradeNow = new QLabel("Upgrade now");
  vlayout->addWidget(upgradeNow);
  
  QLabel *description = new QLabel("Become a comma prime member in the comma app and get premium features!");
  description->setStyleSheet(R"(
    font-size: 50px;
    color: #b8b8b8;
  )");
  description->setWordWrap(true);
  vlayout->addWidget(description);

  vlayout->addSpacing(50);
  
  QVector<QString> features = {"✓ REMOTE ACCESS", "✓ 14 DAYS OF STORAGE", "✓ DEVELOPER PERKS"};
  for (auto featureContent : features){
    QLabel *feature = new QLabel(featureContent);
    feature->setStyleSheet(R"(
      font-size: 40px;
    )");

    vlayout->addWidget(feature);
    vlayout->addSpacing(15);
  }

  setLayout(vlayout);
}


SetupWidget::SetupWidget(QWidget *parent) : QWidget(parent){
  api = new CommaApi(this);

  QVBoxLayout *backgroundLayout = new QVBoxLayout;

  backgroundLayout->addSpacing(100);

  QFrame *background = new QFrame;

  mainLayout = new QStackedLayout;
  
  QWidget *blankWidget = new QWidget;
  mainLayout->addWidget(blankWidget);

  QVBoxLayout *qrLayout = new QVBoxLayout;
  
  QLabel *qrLabel = new QLabel("Pair with Comma Connect app!");
  qrLabel->setStyleSheet(R"(
    font-size: 50px;
  )");
  qrLayout->addWidget(qrLabel);
  
  qrLayout->addWidget(new PairingQRWidget);

  QWidget *q = new QWidget;
  q->setLayout(qrLayout);
  mainLayout->addWidget(q);

  PrimeAdWidget *primeAd = new PrimeAdWidget();
  mainLayout->addWidget(primeAd);

  PrimeUserWidget *primeUserWidget = new PrimeUserWidget();
  mainLayout->addWidget(primeUserWidget);

  background->setLayout(mainLayout);
  background->setStyleSheet(R"(
    .QFrame {
      border-radius: 40px;
      padding: 60px;
    }
  )");
  backgroundLayout->addWidget(background);
  setLayout(backgroundLayout);

  QTimer *timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(1000);
}

void SetupWidget::refresh(){
  QString token = api->create_jwt();

  QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/" + dongle_id + "/"));
  request.setRawHeader("Authorization", ("JWT "+token).toUtf8());
  if(reply == NULL){
    reply = api->get(request);
    connect(reply, &QNetworkReply::finished, this, &SetupWidget::replyFinished);
  }else{
    qDebug()<<"Too many requests, previous request was not yet removed";
  }
}

void SetupWidget::replyFinished() {
  QString answer = reply->readAll();

  QJsonDocument doc = QJsonDocument::fromJson(answer.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting pairing and prime status";
    reply->deleteLater();
    reply = NULL;
    return;
  }
  if(mainLayout->currentIndex() == 0){// If we are still on the blank widget
    setStyleSheet(R"(
      font-size: 90px;
      font-weight: bold;
      background-color: #292929;
    )");
  }
  QJsonObject json = doc.object();
  bool is_paired = json["is_paired"].toBool();
  bool is_prime = json["prime"].toBool();

  if (!is_paired) {
    mainLayout->setCurrentIndex(1);
  }else if (is_paired && !is_prime) {
    mainLayout->setCurrentIndex(2);
  }else if (is_paired && is_prime) {
    mainLayout->setCurrentIndex(3);
  }
  reply->deleteLater();
  reply = NULL;// Make room for new reply
}
