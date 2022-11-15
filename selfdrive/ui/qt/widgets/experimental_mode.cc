#include "selfdrive/ui/qt/widgets/experimental_mode.h"

#include <QDebug>
#include <QLayout>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"

ExperimentalMode::ExperimentalMode(QWidget *parent) : QPushButton(parent) {
//  QPixmap pixmap("../assets/offroad/icon_settings.png");
  QPixmap pixmap("../assets/icon_experimental.png");
  setIcon(pixmap);
  setLayoutDirection(Qt::RightToLeft);
  setIconSize({75, 75});
  connect(this, &QPushButton::clicked, [=]() { emit openSettings(2); });  // show toggles

  setStyleSheet(R"(
    QPushButton {
      font-size: 45px;
      font-weight: 300;
      border: none;
      text-align: left;
      padding: 25 50;
      font-family: JetBrainsMono;
      border-radius: 10px;
      color: #000000;
    }

    QPushButton[experimental_mode="false"] {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #2395FF, stop:1 #14FFAB);
    }
    QPushButton[experimental_mode="false"]:pressed {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #CC2395FF, stop:1 #CC14FFAB);
    }

    QPushButton[experimental_mode="true"] {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #FF9B3F, stop:1 #DB3822);
    }
    QPushButton[experimental_mode="true"]:pressed {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #CCFF9B3F, stop:1 #CCDB3822);
    }
  )");
}

void ExperimentalMode::showEvent(QShowEvent *event) {
  qDebug() << "showEvent!";
//  experimental_mode = params.getBool("ExperimentalMode");
  setProperty("experimental_mode", params.getBool("ExperimentalMode"));
  qDebug() << "experimental_mode" << experimental_mode;
  update();
  setText(experimental_mode ? "MODE: EXPERIMENTAL" : "MODE: CHILL");
  updateStyle();
//  style()->unpolish(this);
//  style()->polish(this);
}
