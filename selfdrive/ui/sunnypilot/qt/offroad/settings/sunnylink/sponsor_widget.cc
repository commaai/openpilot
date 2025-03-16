/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/sunnylink/sponsor_widget.h"

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/api.h"
#include "selfdrive/ui/sunnypilot/qt/util.h"
#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/sunnylink_client.h"

// Sponsor Upsell
using qrcodegen::QrCode;

SunnylinkSponsorQRWidget::SunnylinkSponsorQRWidget(bool sponsor_pair, QWidget* parent) : QWidget(parent), sponsor_pair(sponsor_pair) {
  timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &SunnylinkSponsorQRWidget::refresh);
}

void SunnylinkSponsorQRWidget::showEvent(QShowEvent *event) {
  refresh();
  timer->start(5 * 60 * 1000);
  device()->setOffroadBrightness(100);
}

void SunnylinkSponsorQRWidget::hideEvent(QHideEvent *event) {
  timer->stop();
  device()->setOffroadBrightness(BACKLIGHT_OFFROAD);
}

void SunnylinkSponsorQRWidget::refresh() {
  QString qrString;

  if (sponsor_pair) {
    QString token = SunnylinkApi::create_jwt({}, 3600, true);
    auto sl_dongle_id = getSunnylinkDongleId();
    QByteArray payload = QString("1|" + sl_dongle_id.value_or("") + "|" + token).toUtf8().toBase64();
    qrString = SUNNYLINK_BASE_URL + "/sso?state=" + payload;
  } else {
    qrString = "https://github.com/sponsors/sunnyhaibin";
  }

  this->updateQrCode(qrString);
  update();
}

void SunnylinkSponsorQRWidget::updateQrCode(const QString &text) {
  QrCode qr = QrCode::encodeText(text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  QImage im(sz, sz, QImage::Format_RGB32);

  QRgb black = qRgb(0, 0, 0);
  QRgb white = qRgb(255, 255, 255);
  for (int y = 0; y < sz; y++) {
    for (int x = 0; x < sz; x++) {
      im.setPixel(x, y, qr.getModule(x, y) ? black : white);
    }
  }

  // Integer division to prevent anti-aliasing
  int final_sz = ((width() / sz) - 1) * sz;
  img = QPixmap::fromImage(im.scaled(final_sz, final_sz, Qt::KeepAspectRatio), Qt::MonoOnly);
}

void SunnylinkSponsorQRWidget::paintEvent(QPaintEvent *e) {
  QPainter p(this);
  p.fillRect(rect(), Qt::white);

  QSize s = (size() - img.size()) / 2;
  p.drawPixmap(s.width(), s.height(), img);
}

QStringList SunnylinkSponsorPopup::getInstructions(bool sponsor_pair) {
  QStringList instructions;
  if (sponsor_pair) {
    instructions << tr("Scan the QR code to login to your GitHub account")
                 << tr("Follow the prompts to complete the pairing process")
                 << tr("Re-enter the \"sunnylink\" panel to verify sponsorship status")
                 << tr("If sponsorship status was not updated, please contact a moderator on Discord at https://discord.gg/sunnypilot");
  } else {
    instructions << tr("Scan the QR code to visit sunnyhaibin's GitHub Sponsors page")
                 << tr("Choose your sponsorship tier and confirm your support")
                 << tr("Join our community on Discord at https://discord.gg/sunnypilot and reach out to a moderator to confirm your sponsor status");
  }
  return instructions;
}

SunnylinkSponsorPopup::SunnylinkSponsorPopup(bool sponsor_pair, QWidget *parent) : DialogBase(parent), sponsor_pair(sponsor_pair) {
  auto *hlayout = new QHBoxLayout(this);
  auto sunnylink_client = new SunnylinkClient(this);
  hlayout->setContentsMargins(0, 0, 0, 0);
  hlayout->setSpacing(0);

  setStyleSheet("SunnylinkSponsorPopup { background-color: #E0E0E0; }");

  // text
  auto vlayout = new QVBoxLayout();
  vlayout->setContentsMargins(85, 70, 50, 70);
  vlayout->setSpacing(50);
  hlayout->addLayout(vlayout, 1);
  {
    auto close = new QPushButton(QIcon(":/icons/close.svg"), "", this);
    close->setIconSize(QSize(80, 80));
    close->setStyleSheet("border: none;");
    vlayout->addWidget(close, 0, Qt::AlignLeft);
    connect(close, &QPushButton::clicked, this, [=] {
      sunnylink_client->role_service->load();
      sunnylink_client->user_service->load();
      QDialog::reject();
    });

    //vlayout->addSpacing(30);

    const QString titleText = sponsor_pair ? tr("Pair your GitHub account") : tr("Early Access: Become a sunnypilot Sponsor");
    const auto title = new QLabel(titleText, this);
    title->setStyleSheet("font-size: 75px; color: black;");
    title->setWordWrap(true);
    vlayout->addWidget(title);

    QStringList instructions = getInstructions(sponsor_pair);
    QString instructionsHtml = "<ol type='1' style='margin-left: 15px;'>";
    for (const auto & instruction : instructions) {
      instructionsHtml += QString("<li style='margin-bottom: 50px;'>%1</li>").arg(instruction);
    }
    instructionsHtml += "</ol>";
    const auto instructionsLabel = new QLabel(instructionsHtml, this);


    instructionsLabel->setStyleSheet("font-size: 47px; font-weight: bold; color: black;");
    instructionsLabel->setWordWrap(true);
    vlayout->addWidget(instructionsLabel);

    vlayout->addStretch();
  }

  // QR code
  auto *qr = new SunnylinkSponsorQRWidget(sponsor_pair, this);
  hlayout->addWidget(qr, 1);
}
