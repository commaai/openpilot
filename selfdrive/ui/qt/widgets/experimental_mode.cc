#include "selfdrive/ui/qt/widgets/experimental_mode.h"

#include <QDebug>
#include <QLayout>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"

ExperimentalMode::ExperimentalMode(QWidget *parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 0, 50, 0);
  button = new QPushButton;

  QPixmap pixmap("../assets/offroad/icon_settings.png");
  button->setIcon(pixmap);
//  button->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  button->setLayoutDirection(Qt::RightToLeft);
  qDebug() << pixmap.rect().size();
  button->setIconSize({75, 75});
//  const QRect settings_btn = QRect(50, 35, 200, 117);
//  QPixmap settings_img = loadPixmap("../../assets/images/button_settings.png", settings_btn.size(), Qt::IgnoreAspectRatio);

//  main_layout->addWidget(label);
//  QLayout *layout = new QLayout(this);
//  layout->setContentsMargins(0, 0, 0, 0);
  main_layout->addWidget(button);

  setStyleSheet(R"(
    QPushButton {
      font-size: 45px;
      font-weight: 300;
      border: none;
      text-align: left;
      padding-top: 0px;
      padding-bottom: 0px;
      font-family: JetBrainsMono;

    }

    ExperimentalMode {
      border-radius: 10px;
      background: #000000;
    }
    ExperimentalMode:pressed {
      border-radius: 50px;
      background: #ffffff;
      color: #000000;
    }
    ExperimentalMode[experimental_mode="false"] {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #2395FF, stop:1 #14FFAB);
    }
    ExperimentalMode[experimental_mode="true"] {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #FF9B3F, stop:1 #DB3822);
    }
  )");
}

void ExperimentalMode::showEvent(QShowEvent *event) {
  qDebug() << "showEvent!";
//  experimental_mode = params.getBool("ExperimentalMode");
  setProperty("experimental_mode", params.getBool("ExperimentalMode"));
  qDebug() << "experimental_mode" << experimental_mode;
  update();
  button->setText(experimental_mode ? "MODE: EXPERIMENTAL" : "MODE: CHILL");
  updateStyle();
//  style()->unpolish(this);
//  style()->polish(this);
}
