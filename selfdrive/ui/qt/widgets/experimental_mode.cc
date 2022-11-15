#include "selfdrive/ui/qt/widgets/experimental_mode.h"

#include <QDebug>
#include <QHBoxLayout>
#include <QStyle>

#include "selfdrive/ui/ui.h"

ExperimentalModeButton::ExperimentalModeButton(QWidget *parent) : QPushButton(parent) {
  chill_pixmap = QPixmap("../assets/img_couch.png").scaledToWidth(100, Qt::SmoothTransformation);
  experimental_pixmap = QPixmap("../assets/img_experimental_grey.png").scaledToWidth(100, Qt::SmoothTransformation);

  setFixedHeight(125);
  connect(this, &QPushButton::clicked, [=]() { emit openSettings(2); });  // show toggles

  QWidget *verticalLine = new QWidget;
  verticalLine->setFixedWidth(3);
  verticalLine->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  verticalLine->setStyleSheet(QString("background-color: #4D000000;"));

  QHBoxLayout *main_layout = new QHBoxLayout;
  main_layout->setContentsMargins(30, 0, 30, 0);

  mode_label = new QLabel;
  mode_icon = new QLabel;
  mode_icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));

  main_layout->addWidget(mode_label, 1, Qt::AlignLeft);
  main_layout->addWidget(verticalLine, 0, Qt::AlignRight);
  main_layout->addSpacing(30);
  main_layout->addWidget(mode_icon, 0, Qt::AlignRight);

  setLayout(main_layout);

  setStyleSheet(R"(
    QPushButton {
      border: none;
      padding: 0 50;
      border-radius: 10px;
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

void ExperimentalModeButton::showEvent(QShowEvent *event) {
  ui_update_params(uiState());
  setProperty("experimental_mode", uiState()->scene.experimental_mode);
  mode_icon->setPixmap(experimental_mode ? experimental_pixmap : chill_pixmap);
  mode_label->setText(experimental_mode ? tr("EXPERIMENTAL MODE ON") : tr("CHILL MODE ON"));
  style()->unpolish(this);
  style()->polish(this);
}
