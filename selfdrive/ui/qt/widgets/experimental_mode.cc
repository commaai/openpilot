#include "selfdrive/ui/qt/widgets/experimental_mode.h"

#include <QDebug>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"

ExperimentalMode::ExperimentalMode(QWidget *parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 50, 50, 60);
  label = new QLabel;
  button = new QPushButton;
  label->setProperty("type", "title");
//  main_layout->addWidget(label);
  main_layout->addWidget(button);

  setStyleSheet(R"(
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

    QLabel[type="title"] { font-size: 51px; font-weight: 500; }
    QLabel[type="number"] { font-size: 78px; font-weight: 500; }
    QLabel[type="unit"] { font-size: 51px; font-weight: 300; color: #A0A0A0; }
  )");
}

void ExperimentalMode::showEvent(QShowEvent *event) {
  qDebug() << "showEvent!";
//  experimental_mode = params.getBool("ExperimentalMode");
  setProperty("experimental_mode", params.getBool("ExperimentalMode"));
  qDebug() << "experimental_mode" << experimental_mode;
  update();
  label->setText(experimental_mode ? "MODE: EXPERIMENTAL" : "MODE: CHILL");
  updateStyle();
//  style()->unpolish(this);
//  style()->polish(this);
}
