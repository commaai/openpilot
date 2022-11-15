#include "selfdrive/ui/qt/widgets/experimental_mode.h"

#include <QDebug>
#include <QLayout>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"

ExperimentalMode::ExperimentalMode(QWidget *parent) : QPushButton(parent) {
//  setIconSize({100, 100});
//  setLayoutDirection(Qt::RightToLeft);
  setFixedHeight(125);
  connect(this, &QPushButton::clicked, [=]() { emit openSettings(2); });  // show toggles

  experimental_pixmap = QPixmap("../assets/img_experimental_grey.png").scaledToWidth(100, Qt::SmoothTransformation);
  chill_pixmap = QPixmap("../assets/img_couch.png").scaledToWidth(100, Qt::SmoothTransformation);

  mode_icon = new QLabel;
//  mode_icon->setPixmap(chill_pixmap.scaledToWidth(100, Qt::SmoothTransformation));
  mode_icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));


  QWidget *verticalLine = new QWidget;
  verticalLine->setFixedWidth(2);
  verticalLine->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  verticalLine->setStyleSheet(QString("background-color: #4D000000;"));

  QHBoxLayout *main_layout = new QHBoxLayout;
  main_layout->setContentsMargins(30, 0, 30, 0);

  mode_label = new QLabel;

  main_layout->addWidget(mode_label, 1, Qt::AlignLeft);
  main_layout->addWidget(verticalLine, 0, Qt::AlignRight);
  main_layout->addSpacing(30);
  main_layout->addWidget(mode_icon, 0, Qt::AlignRight);

  setLayout(main_layout);

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

    QLabel {
      font-size: 45px;
      font-weight: 300;
      text-align: left;
      font-family: JetBrainsMono;
      color: #000000;
    }

    QPushButton[experimental_mode="false"] {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #14FFAB, stop:1 #2395FF);
    }
    QPushButton[experimental_mode="false"]:pressed {
      background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                  stop:0 #CC14FFAB, stop:1 #CC2395FF);
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
//  auto cp_bytes = params.get("CarParamsPersistent");
//  if (!cp_bytes.empty()) {
//    AlignedBuffer aligned_buf;
//    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
//    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
//    const bool op_long = CP.getOpenpilotLongitudinalControl() && !CP.getExperimentalLongitudinalAvailable();
//    const bool exp_long_enabled = CP.getExperimentalLongitudinalAvailable() && params.getBool("ExperimentalLongitudinalEnabled");
//    if (op_long || exp_long_enabled) {
//      show();
//    } else {
//      hide();
//    }
//  }


  qDebug() << "showEvent!";
//  experimental_mode = params.getBool("ExperimentalMode");
  setProperty("experimental_mode", params.getBool("ExperimentalMode"));
//  setIcon(experimental_mode ? experimental_pixmap : chill_pixmap);
//  setText(experimental_mode ? "EXPERIMENTAL MODE" : "CHILL MODE");
//  mode_icon->setIcon(experimental_mode ? experimental_pixmap : chill_pixmap);
  mode_icon->setPixmap(experimental_mode ? experimental_pixmap : chill_pixmap);
  mode_label->setText(experimental_mode ? "EXPERIMENTAL MODE ON" : "CHILL MODE ON");
  updateStyle();
//  hide();
//  QHBoxLayout *layout = new QHBoxLayout;
//  layout->addWidget(new QLabel("hia"));
//  layout->addWidget(new QLabel("hia2"));
//  setLayout(layout);
//  style()->unpolish(this);
//  style()->polish(this);
}
