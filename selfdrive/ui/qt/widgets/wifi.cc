#include "selfdrive/ui/qt/widgets/wifi.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QStackedLayout>

WiFiPromptWidget::WiFiPromptWidget(QWidget *parent) : QFrame(parent) {
  QStackedLayout *stack = new QStackedLayout(this);

  // Setup Wi-Fi
  QFrame *setup = new QFrame;
  QVBoxLayout *setup_layout = new QVBoxLayout(setup);
  setup_layout->setContentsMargins(64, 48, 64, 48);
  setup_layout->setSpacing(32);
  {
    QHBoxLayout *title_layout = new QHBoxLayout;
    title_layout->setSpacing(32);
    {
      QLabel *icon = new QLabel;
      QPixmap *pixmap = new QPixmap("../assets/offroad/icon_wifi_strength_full.svg");
      icon->setPixmap(pixmap->scaledToWidth(80, Qt::SmoothTransformation));
      title_layout->addWidget(icon);

      QLabel *title = new QLabel(tr("Setup Wi-Fi"));
      title->setStyleSheet("font-size: 64px; font-weight: 600;");
      title_layout->addWidget(title);
      title_layout->addStretch();
    }
    setup_layout->addLayout(title_layout);

    QLabel *desc = new QLabel(tr("Connect to Wi-Fi to upload driving data and help improve openpilot"));
    desc->setStyleSheet("font-size: 36px; font-weight: 400;");
    desc->setWordWrap(true);
    setup_layout->addWidget(desc);

    QPushButton *settings_btn = new QPushButton(tr("Open Settings"));
    settings_btn->setStyleSheet(R"(
      QPushButton {
        font-size: 48px;
        font-weight: 500;
        border-radius: 10px;
        background-color: #465BEA;
        padding: 32px;
      }
      QPushButton:pressed {
        background-color: #3049F4;
      }
    )");
    setup_layout->addWidget(settings_btn, 0);

    setup_layout->addStretch();
  }
  stack->addWidget(setup);

  // Uploading data
  QFrame *uploading = new QFrame;
  QVBoxLayout *uploading_layout = new QVBoxLayout(uploading);
  uploading_layout->setContentsMargins(40, 40, 40, 40);
  {
    QLabel *icon = new QLabel;
    QPixmap *pixmap = new QPixmap("../assets/offroad/icon_wifi_uploading.svg");
    icon->setPixmap(pixmap->scaledToWidth(100, Qt::SmoothTransformation));
    uploading_layout->addWidget(icon, 0, Qt::AlignHCenter);

    QLabel *title = new QLabel(tr("Uploading your training data"));
    title->setAlignment(Qt::AlignHCenter);
    title->setStyleSheet("font-size: 64px; font-weight: 600;");
    title->setWordWrap(true);
    uploading_layout->addWidget(title, 0, Qt::AlignHCenter);

    QLabel *desc = new QLabel(tr("Your driving data helps improve openpilot"));
    desc->setStyleSheet("font-size: 34px; font-weight: 400;");
    desc->setWordWrap(false);
    uploading_layout->addWidget(desc, 0, Qt::AlignHCenter);
  }
  stack->addWidget(uploading);

  stack->setCurrentWidget(uploading);

  setFixedWidth(750);
  setStyleSheet(R"(
    WiFiPromptWidget {
      background-color: #333333;
      border-radius: 10px;
    }
  )");
}
