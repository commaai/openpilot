#include "selfdrive/ui/qt/widgets/wifi.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>

WiFiPromptWidget::WiFiPromptWidget(QWidget *parent) : QFrame(parent) {
  // Setup Firehose Mode
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(56, 40, 56, 40);
  main_layout->setSpacing(42);  
  
  QLabel *title = new QLabel(tr("<span style='font-family: \"Noto Color Emoji\";'>🔥</span> Firehose Mode <span style='font-family: Noto Color Emoji;'>🔥</span>"));
  title->setStyleSheet("font-size: 64px; font-weight: 500;");
  main_layout->addWidget(title);

  QLabel *desc = new QLabel(tr("Maximize your training data uploads to improve openpilot's driving models."));
  desc->setStyleSheet("font-size: 40px; font-weight: 400;");
  desc->setWordWrap(true);
  main_layout->addWidget(desc);

  QPushButton *settings_btn = new QPushButton(tr("Open"));
  connect(settings_btn, &QPushButton::clicked, [=]() { emit openSettings(1, "FirehosePanel"); });
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
  main_layout->addWidget(settings_btn);

  setStyleSheet(R"(
    WiFiPromptWidget {
      background-color: #333333;
      border-radius: 10px;
    }
  )");
}