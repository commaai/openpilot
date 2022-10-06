#include "tools/cabana/videowidget.h"

#include <QButtonGroup>
#include <QDateTime>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>

#include "tools/cabana/parser.h"

inline QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh::mm::ss" : "mm::ss");
}

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  cam_widget = new CameraViewWidget("camerad", VISION_STREAM_ROAD, false, this);
  cam_widget->setFixedSize(parent->width(), parent->width() / 1.596);
  main_layout->addWidget(cam_widget);

  // slider controls
  QHBoxLayout *slider_layout = new QHBoxLayout();
  time_label = new QLabel("00:00");
  slider_layout->addWidget(time_label);

  slider = new QSlider(Qt::Horizontal, this);
  QObject::connect(slider, &QSlider::sliderMoved, [=]() {
    time_label->setText(formatTime(slider->value()));
  });
  slider->setSingleStep(1);
  slider->setMaximum(parser->replay->totalSeconds());
  QObject::connect(slider, &QSlider::sliderReleased, [=]() {
    time_label->setText(formatTime(slider->value()));
    parser->replay->seekTo(slider->value(), false);
  });
  slider_layout->addWidget(slider);

  total_time_label = new QLabel(formatTime(parser->replay->totalSeconds()));
  slider_layout->addWidget(total_time_label);

  main_layout->addLayout(slider_layout);

  // btn controls
  QHBoxLayout *control_layout = new QHBoxLayout();
  QPushButton *play = new QPushButton("⏸");
  play->setStyleSheet("font-weight:bold");
  QObject::connect(play, &QPushButton::clicked, [=]() {
    bool is_paused = parser->replay->isPaused();
    play->setText(is_paused ? "⏸" : "▶");
    parser->replay->pause(!is_paused);
  });
  control_layout->addWidget(play);

  QButtonGroup *group = new QButtonGroup(this);
  group->setExclusive(true);
  for (float speed : {0.1, 0.5, 1., 2.}) {
    QPushButton *btn = new QPushButton(QString("%1x").arg(speed), this);
    btn->setCheckable(true);
    QObject::connect(btn, &QPushButton::clicked, [=]() { parser->replay->setSpeed(speed); });
    control_layout->addWidget(btn);
    group->addButton(btn);
    if (speed == 1.0) btn->setChecked(true);
  }

  main_layout->addLayout(control_layout);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

  QObject::connect(parser, &Parser::rangeChanged, this, &VideoWidget::rangeChanged);
  QObject::connect(parser, &Parser::updated, this, &VideoWidget::updateState);
}

void VideoWidget::rangeChanged(double min, double max) {
  if (!parser->isZoomed()) {
    min = 0;
    max = parser->replay->totalSeconds();
  }
  time_label->setText(formatTime(min));
  total_time_label->setText(formatTime(max));
  slider->setMaximum(max);
  slider->setValue(parser->currentSec());
}

void VideoWidget::updateState() {
  if (!slider->isSliderDown()) {
    int current_sec = parser->currentSec();
    time_label->setText(formatTime(current_sec));
    slider->setValue(current_sec);
  }
}
