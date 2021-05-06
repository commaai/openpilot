#include "selfdrive/ui/qt/onroad/hud.h"

StatusIcon::StatusIcon(const QString &path, QWidget *parent) : QWidget(parent) {
  setFixedSize(radius, radius);
  img = QImage(path).scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void StatusIcon::setBackground(const QColor bg, const float op) {
  if (bg != background || op != opacity) {
    background = bg;
    opacity = op;
    repaint();
  }
}

void StatusIcon::paintEvent(QPaintEvent *e) {
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setBrush(QBrush(background));
  p.drawEllipse(rect());
  p.setOpacity(opacity);
  p.drawImage((width() - img.width()) / 2, (height() - img.height()) / 2, img);
}

VisionOverlay::VisionOverlay(QWidget *parent) : QWidget(parent) {
  // ***** header *****
  QHBoxLayout *header = new QHBoxLayout();
  header->setMargin(0);
  header->setSpacing(0);

  // max speed
  QVBoxLayout *maxspeed_layout = new QVBoxLayout();
  maxspeed_layout->setMargin(20);

  QLabel *max = new QLabel("MAX");
  max->setAlignment(Qt::AlignCenter);
  max->setStyleSheet("font-size: 40px; font-weight: 400;");
  maxspeed_layout->addWidget(max, 0, Qt::AlignTop);

  maxspeed = new QLabel();
  maxspeed->setAlignment(Qt::AlignCenter);
  maxspeed->setStyleSheet("font-size: 78px; font-weight: 500;");
  maxspeed_layout->addWidget(maxspeed, 0, Qt::AlignCenter);

  QWidget *ms = new QWidget();
  ms->setFixedSize(180, 200);
  ms->setObjectName("MaxSpeedContainer");
  ms->setLayout(maxspeed_layout);
  ms->setStyleSheet(R"(
    #MaxSpeedContainer {
      border-width: 8px;
      border-style: solid;
      border-radius: 20px;
      border-color: rgba(255, 255, 255, 100);
      background-color: rgba(0, 0, 0, 100);
    }
  )");

  header->addWidget(ms, 0, Qt::AlignLeft | Qt::AlignTop);

  // current speed
  QVBoxLayout *speed_layout = new QVBoxLayout();
  speed_layout->setMargin(0);
  speed_layout->setSpacing(0);
  header->addLayout(speed_layout, 1);

  speed = new QLabel();
  speed->setStyleSheet("font-size: 180px; font-weight: 500;");
  speed_layout->addWidget(speed, 0, Qt::AlignHCenter);

  speed_unit = new QLabel();
  speed_unit->setStyleSheet("font-size: 70px; font-weight: 400; color: rgba(255, 255, 255, 200);");
  speed_layout->addWidget(speed_unit, 0, Qt::AlignHCenter | Qt::AlignTop);

  // engage-ability icon
  wheel = new StatusIcon("../assets/img_chffr_wheel.png");
  header->addWidget(wheel, 0, Qt::AlignRight | Qt::AlignTop);


  // ***** footer *****
  QHBoxLayout *footer = new QHBoxLayout();
  footer->setMargin(0);
  footer->setSpacing(0);

  // DM icon
  monitoring = new StatusIcon("../assets/img_driver_face.png", this);
  footer->addWidget(monitoring, 0, Qt::AlignLeft);

  footer->addStretch(1);

  // build container layout
  layout = new QVBoxLayout();
  layout->setMargin(40);
  layout->addLayout(header);
  layout->addStretch(1);
  layout->addLayout(footer);

  setLayout(layout);
  setStyleSheet("color: white;");
}

void VisionOverlay::update(const UIState &s) {
  auto v = s.scene.car_state.getVEgo() * (s.scene.is_metric ? 3.6 : 2.2369363);
  speed->setText(QString::number((int)v));
  speed_unit->setText(s.scene.is_metric ? "km/h" : "mph");

  // hack: remove ascent + descent
  if (speed->minimumHeight() != speed->maximumHeight()) {
    QFontMetrics fm(speed->font());
    speed->setFixedHeight(fm.tightBoundingRect("0123456789").height());
  }

  const int SET_SPEED_NA = 255;
  auto vcruise = s.scene.controls_state.getVCruise();
  if (vcruise != 0 && vcruise != SET_SPEED_NA) {
    auto max = vcruise * (s.scene.is_metric ? 1 : 0.6225);
    maxspeed->setText(QString::number((int)max));
  } else {
    maxspeed->setText("N/A");
  }

  float dm_alpha = s.scene.dmonitoring_state.getIsActiveMode() ? 1.0 : 0.2;
  float alert_visible = s.scene.controls_state.getAlertSize() != cereal::ControlsState::AlertSize::NONE;
  monitoring->setBackground(QColor(0, 0, 0, 70), alert_visible ? 0.0 : dm_alpha);

  float wheel_alpha = s.scene.controls_state.getEngageable() ? 1.0 : 0.0;
  wheel->setBackground(bg_colors[s.status], wheel_alpha);
}
