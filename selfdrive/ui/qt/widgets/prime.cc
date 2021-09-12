#include "selfdrive/ui/qt/widgets/prime.h"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QrCode.hpp>

#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"

using qrcodegen::QrCode;

PairingQRWidget::PairingQRWidget(QWidget* parent) : QWidget(parent) {
  qrCode = new QLabel;
  qrCode->setScaledContents(true);
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->addWidget(qrCode, 0, Qt::AlignCenter);

  QTimer* timer = new QTimer(this);
  timer->start(30 * 1000);
  connect(timer, &QTimer::timeout, this, &PairingQRWidget::refresh);
}

void PairingQRWidget::showEvent(QShowEvent *event) {
  refresh();
}

void PairingQRWidget::refresh() {
  QString pairToken = CommaApi::create_jwt({{"pair", true}});
  QString qrString = "https://connect.comma.ai/?pair=" + pairToken;
  this->updateQrCode(qrString);
}

void PairingQRWidget::updateQrCode(const QString &text) {
  QrCode qr = QrCode::encodeText(text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  // make the image larger so we can have a white border
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
  mainLayout = new QVBoxLayout(this);
  mainLayout->setMargin(0);
  mainLayout->setSpacing(30);

  // subscribed prime layout
  QWidget *primeWidget = new QWidget;
  primeWidget->setObjectName("primeWidget");
  QVBoxLayout *primeLayout = new QVBoxLayout(primeWidget);
  primeLayout->setMargin(0);
  primeWidget->setContentsMargins(60, 50, 60, 50);

  QLabel* subscribed = new QLabel("✓ SUBSCRIBED");
  subscribed->setStyleSheet("font-size: 41px; font-weight: bold; color: #86FF4E;");
  primeLayout->addWidget(subscribed, 0, Qt::AlignTop);

  primeLayout->addSpacing(60);

  QLabel* commaPrime = new QLabel("comma prime");
  commaPrime->setStyleSheet("font-size: 75px; font-weight: bold;");
  primeLayout->addWidget(commaPrime, 0, Qt::AlignTop);

  primeLayout->addSpacing(20);

  QLabel* connectUrl = new QLabel("CONNECT.COMMA.AI");
  connectUrl->setStyleSheet("font-size: 41px; font-family: Inter SemiBold; color: #A0A0A0;");
  primeLayout->addWidget(connectUrl, 0, Qt::AlignTop);

  mainLayout->addWidget(primeWidget);

  // comma points layout
  QWidget *pointsWidget = new QWidget;
  pointsWidget->setObjectName("primeWidget");
  QVBoxLayout *pointsLayout = new QVBoxLayout(pointsWidget);
  pointsLayout->setMargin(0);
  pointsWidget->setContentsMargins(60, 50, 60, 50);

  QLabel* commaPoints = new QLabel("COMMA POINTS");
  commaPoints->setStyleSheet("font-size: 41px; font-family: Inter SemiBold;");
  pointsLayout->addWidget(commaPoints, 0, Qt::AlignTop);

  points = new QLabel("210");
  points->setStyleSheet("font-size: 91px; font-weight: bold;");
  pointsLayout->addWidget(points, 0, Qt::AlignTop);

  mainLayout->addWidget(pointsWidget);

  mainLayout->addStretch();

  // set up API requests
  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1/devices/" + *dongleId + "/owner";
    RequestRepeater *repeater = new RequestRepeater(this, url, "ApiCache_Owner", 6);
    QObject::connect(repeater, &RequestRepeater::receivedResponse, this, &PrimeUserWidget::replyFinished);
  }
}

void PrimeUserWidget::replyFinished(const QString &response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting points";
    return;
  }

  QJsonObject json = doc.object();
  points->setText(QString::number(json["points"].toInt()));
}

PrimeAdWidget::PrimeAdWidget(QWidget* parent) : QFrame(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(80, 90, 80, 60);
  main_layout->setSpacing(0);

  QLabel *upgrade = new QLabel("Upgrade Now");
  upgrade->setStyleSheet("font-size: 75px; font-weight: bold;");
  main_layout->addWidget(upgrade, 0, Qt::AlignTop);
  main_layout->addSpacing(50);

  QLabel *description = new QLabel("Become a comma prime member at connect.comma.ai");
  description->setStyleSheet("font-size: 60px; font-weight: light; color: white;");
  description->setWordWrap(true);
  main_layout->addWidget(description, 0, Qt::AlignTop);

  main_layout->addStretch();

  QLabel *features = new QLabel("PRIME FEATURES:");
  features->setStyleSheet("font-size: 41px; font-weight: bold; color: #E5E5E5;");
  main_layout->addWidget(features, 0, Qt::AlignBottom);
  main_layout->addSpacing(30);

  QVector<QString> bullets = {"Remote access", "14 days of storage", "Developer perks"};
  for (auto &b: bullets) {
    const QString check = "<b><font color='#465BEA'>✓</font></b> ";
    QLabel *l = new QLabel(check + b);
    l->setAlignment(Qt::AlignLeft);
    l->setStyleSheet("font-size: 50px; margin-bottom: 15px;");
    main_layout->addWidget(l, 0, Qt::AlignBottom);
  }

  setStyleSheet(R"(
    PrimeAdWidget {
      border-radius: 10px;
      background-color: #333333;
    }
  )");
}


SetupWidget::SetupWidget(QWidget* parent) : QFrame(parent) {
  mainLayout = new QStackedWidget;

  // Unpaired, registration prompt layout

  QWidget* finishRegistration = new QWidget;
  finishRegistration->setObjectName("primeWidget");
  QVBoxLayout* finishRegistationLayout = new QVBoxLayout(finishRegistration);
  finishRegistationLayout->setContentsMargins(30, 75, 30, 45);
  finishRegistationLayout->setSpacing(0);

  QLabel* registrationTitle = new QLabel("Finish Setup");
  registrationTitle->setStyleSheet("font-size: 75px; font-weight: bold; margin-left: 55px;");
  finishRegistationLayout->addWidget(registrationTitle);

  finishRegistationLayout->addSpacing(30);

  QLabel* registrationDescription = new QLabel("Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer.");
  registrationDescription->setWordWrap(true);
  registrationDescription->setStyleSheet("font-size: 55px; font-weight: light; margin-left: 55px;");
  finishRegistationLayout->addWidget(registrationDescription);

  finishRegistationLayout->addStretch();

  QPushButton* finishButton = new QPushButton("Pair device");
  finishButton->setFixedHeight(220);
  finishButton->setStyleSheet(R"(
    QPushButton {
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #465BEA;
    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )");
  finishRegistationLayout->addWidget(finishButton);
  QObject::connect(finishButton, &QPushButton::clicked, this, &SetupWidget::showQrCode);

  mainLayout->addWidget(finishRegistration);

  // Pairing QR code layout

  QWidget* q = new QWidget;
  q->setObjectName("primeWidget");
  QVBoxLayout* qrLayout = new QVBoxLayout(q);
  qrLayout->setContentsMargins(90, 90, 90, 90);

  QLabel* qrLabel = new QLabel("Scan the QR code to pair.");
  qrLabel->setAlignment(Qt::AlignHCenter);
  qrLabel->setStyleSheet("font-size: 47px; font-weight: light;");
  qrLayout->addWidget(qrLabel);
  qrLayout->addSpacing(50);

  qrLayout->addWidget(new PairingQRWidget);
  qrLayout->addStretch();

  // setup widget
  QVBoxLayout *outer_layout = new QVBoxLayout(this);
  outer_layout->setContentsMargins(0, 0, 0, 0);
  outer_layout->addWidget(mainLayout);

  mainLayout->addWidget(q);

  primeAd = new PrimeAdWidget;
  mainLayout->addWidget(primeAd);

  primeUser = new PrimeUserWidget;
  mainLayout->addWidget(primeUser);

  mainLayout->setCurrentWidget(primeAd);

  setFixedWidth(750);
  setStyleSheet(R"(
    #primeWidget {
      border-radius: 10px;
      background-color: #333333;
    }
  )");

  // Retain size while hidden
  QSizePolicy sp_retain = sizePolicy();
  sp_retain.setRetainSizeWhenHidden(true);
  setSizePolicy(sp_retain);

  // set up API requests
  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1.1/devices/" + *dongleId + "/";
    RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_Device", 5);

    QObject::connect(repeater, &RequestRepeater::receivedResponse, this, &SetupWidget::replyFinished);
    QObject::connect(repeater, &RequestRepeater::failedResponse, this, &SetupWidget::parseError);
  }
  hide(); // Only show when first request comes back
}

void SetupWidget::parseError(const QString &response) {
  show();
  if (mainLayout->currentIndex() == 1) {
    showQr = false;
    mainLayout->setCurrentIndex(0);
  }
}

void SetupWidget::showQrCode() {
  showQr = true;
  mainLayout->setCurrentIndex(1);
}

void SetupWidget::replyFinished(const QString &response) {
  show();
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting pairing and prime status";
    return;
  }

  QJsonObject json = doc.object();
  if (!json["is_paired"].toBool()) {
    mainLayout->setCurrentIndex(showQr);
  } else if (!json["prime"].toBool()) {
    showQr = false;
    mainLayout->setCurrentWidget(primeAd);
  } else {
    showQr = false;
    mainLayout->setCurrentWidget(primeUser);
  }
}
