#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QStackedLayout>
#include <QTimer>
#include <QVBoxLayout>

#include "QrCode.hpp"
#include "api.hpp"
#include "common/params.h"
#include "setup.hpp"

using qrcodegen::QrCode;

PairingQRWidget::PairingQRWidget(QWidget* parent) : QWidget(parent) {
  qrCode = new QLabel;
  qrCode->setScaledContents(true);
  QVBoxLayout* v = new QVBoxLayout;
  v->addWidget(qrCode, 0, Qt::AlignCenter);
  setLayout(v);

  QTimer* timer = new QTimer(this);
  timer->start(30 * 1000);
  connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  refresh(); // Not waiting for the first refresh
}

void PairingQRWidget::refresh(){
  QString IMEI = QString::fromStdString(Params().get("IMEI"));
  QString serial = QString::fromStdString(Params().get("HardwareSerial"));

  if (std::min(IMEI.length(), serial.length()) <= 5) {
    qrCode->setText("Error getting serial: contact support");
    qrCode->setWordWrap(true);
    qrCode->setStyleSheet(R"(font-size: 60px;)");
    return;
  }
  QVector<QPair<QString, QJsonValue>> payloads;
  payloads.push_back(qMakePair(QString("pair"), true));
  QString pairToken = CommaApi::create_jwt(payloads);

  QString qrString = IMEI + "--" + serial + "--" + pairToken;
  this->updateQrCode(qrString);
}

void PairingQRWidget::updateQrCode(QString text) {
  QrCode qr = QrCode::encodeText(text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  // We make the image larger so we can have a white border
  QImage im(sz + 2, sz + 2, QImage::Format_RGB32);
  QRgb black = qRgb(0, 0, 0);
  QRgb white = qRgb(255, 255, 255);

  for (int y = 0; y < sz + 2; y++) {
    for (int x = 0; x < sz + 2; x++) {
      im.setPixel(x, y, white);
    }
  }
  for (int y = 0; y < sz; y++) {
    for (int x = 0; x < sz; x++) {
      im.setPixel(x + 1, y + 1, qr.getModule(x, y) ? black : white);
    }
  }
  // Integer division to prevent anti-aliasing
  int approx500 = (500 / (sz + 2)) * (sz + 2);
  qrCode->setPixmap(QPixmap::fromImage(im.scaled(approx500, approx500, Qt::KeepAspectRatio, Qt::FastTransformation), Qt::MonoOnly));
  qrCode->setFixedSize(approx500, approx500);
}

PrimeUserWidget::PrimeUserWidget(QWidget* parent) : QWidget(parent) {
  mainLayout = new QVBoxLayout;
  QLabel* commaPrime = new QLabel("COMMA PRIME");
  commaPrime->setStyleSheet(R"(
    font-size: 60px;
  )");
  mainLayout->addWidget(commaPrime);

  username = new QLabel("");
  mainLayout->addWidget(username);

  mainLayout->addSpacing(200);

  QLabel* commaPoints = new QLabel("COMMA POINTS");
  commaPoints->setStyleSheet(R"(
    font-size: 60px;
    color: #b8b8b8;
  )");
  mainLayout->addWidget(commaPoints);

  points = new QLabel("");
  mainLayout->addWidget(points);

  setLayout(mainLayout);
  QString dongleId = QString::fromStdString(Params().get("DongleId"));
  QString url = "https://api.commadotai.com/v1/devices/" + dongleId + "/owner";
  RequestRepeater* repeater = new RequestRepeater(this, url, 6);

  QObject::connect(repeater, SIGNAL(receivedResponse(QString)), this, SLOT(replyFinished(QString)));
}

void PrimeUserWidget::replyFinished(QString response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting username and points";
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
}

PrimeAdWidget::PrimeAdWidget(QWidget* parent) : QWidget(parent) {
  QVBoxLayout* vlayout = new QVBoxLayout;

  QLabel* upgradeNow = new QLabel("Upgrade now");
  vlayout->addWidget(upgradeNow);

  QLabel* description = new QLabel("Become a comma prime member in the comma app and get premium features!");
  description->setStyleSheet(R"(
    font-size: 50px;
    color: #b8b8b8;
  )");
  description->setWordWrap(true);
  vlayout->addWidget(description);

  vlayout->addSpacing(50);

  QVector<QString> features = {"✓ REMOTE ACCESS", "✓ 14 DAYS OF STORAGE", "✓ DEVELOPER PERKS"};
  for (auto featureContent : features) {
    QLabel* feature = new QLabel(featureContent);
    feature->setStyleSheet(R"(
      font-size: 40px;
    )");

    vlayout->addWidget(feature);
    vlayout->addSpacing(15);
  }

  setLayout(vlayout);
}


SetupWidget::SetupWidget(QWidget* parent) : QWidget(parent) {
  QVBoxLayout* backgroundLayout = new QVBoxLayout;

  backgroundLayout->addSpacing(100);

  QFrame* background = new QFrame;

  mainLayout = new QStackedLayout;

  QWidget* blankWidget = new QWidget;
  mainLayout->addWidget(blankWidget);

  QWidget* finishRegistration = new QWidget;

  QVBoxLayout* finishRegistationLayout = new QVBoxLayout;
  finishRegistationLayout->addSpacing(30);
  QPushButton* finishButton = new QPushButton("Finish registration");
  finishButton->setFixedHeight(200);
  finishButton->setStyleSheet(R"(
    border-radius: 30px;
    font-size: 55px;
    background: #585858;
  )");
  QObject::connect(finishButton, SIGNAL(released()), this, SLOT(showQrCode()));
  finishRegistationLayout->addWidget(finishButton);

  QLabel* registrationDescription = new QLabel("Pair your device with the comma connect app");
  registrationDescription->setStyleSheet(R"(
    font-size: 55px;
    font-weight: 400;
  )");

  registrationDescription->setWordWrap(true);
  finishRegistationLayout->addWidget(registrationDescription);

  finishRegistration->setLayout(finishRegistationLayout);
  mainLayout->addWidget(finishRegistration);

  QVBoxLayout* qrLayout = new QVBoxLayout;

  QLabel* qrLabel = new QLabel("Pair with comma connect!");
  qrLabel->setStyleSheet(R"(font-size: 40px)");
  qrLayout->addWidget(qrLabel);

  qrLayout->addWidget(new PairingQRWidget);

  QWidget* q = new QWidget;
  q->setLayout(qrLayout);
  mainLayout->addWidget(q);

  PrimeAdWidget* primeAd = new PrimeAdWidget;
  mainLayout->addWidget(primeAd);

  PrimeUserWidget* primeUserWidget = new PrimeUserWidget;
  mainLayout->addWidget(primeUserWidget);

  background->setLayout(mainLayout);
  background->setStyleSheet(R"(
    .QFrame {
      border-radius: 40px;
      padding: 40px;
    }
  )");
  backgroundLayout->addWidget(background);
  setLayout(backgroundLayout);

  QString dongleId = QString::fromStdString(Params().get("DongleId"));
  QString url = "https://api.commadotai.com/v1.1/devices/" + dongleId + "/";
  RequestRepeater* repeater = new RequestRepeater(this, url, 5);

  QObject::connect(repeater, SIGNAL(receivedResponse(QString)), this, SLOT(replyFinished(QString)));
  QObject::connect(repeater, SIGNAL(failedResponse(QString)), this, SLOT(parseError(QString)));
}

void SetupWidget::parseError(QString response) {
  showQr = false;
  mainLayout->setCurrentIndex(0);
  setStyleSheet(R"(
    font-size: 90px;
    background-color: #000000;
  )");
}

void SetupWidget::showQrCode(){
  showQr = true;
  mainLayout->setCurrentIndex(2);
}

void SetupWidget::replyFinished(QString response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting pairing and prime status";
    return;
  }

  if (mainLayout->currentIndex() == 0) { // If we are still on the blank widget
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
    mainLayout->setCurrentIndex(1 + showQr);
  } else if (is_paired && !is_prime) {
    showQr = false;
    mainLayout->setCurrentIndex(3);
  } else if (is_paired && is_prime) {
    showQr = false;
    mainLayout->setCurrentIndex(4);
  }
}
