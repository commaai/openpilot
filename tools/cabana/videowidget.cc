#include "tools/cabana/videowidget.h"

#include <QButtonGroup>
#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionSlider>
#include <QTimer>
#include <QVBoxLayout>

inline QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
}

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // TODO: figure out why the CameraViewWidget crashed occasionally.
  cam_widget = new CameraViewWidget("camerad", VISION_STREAM_ROAD, false, this);
  cam_widget->setFixedSize(parent->width(), parent->width() / 1.596);
  main_layout->addWidget(cam_widget);

  // slider controls
  QHBoxLayout *slider_layout = new QHBoxLayout();
  QLabel *time_label = new QLabel("00:00");
  slider_layout->addWidget(time_label);

  slider = new Slider(this);
  slider->setSingleStep(0);
  slider->setMinimum(0);
  slider->setMaximum(can->totalSeconds() * 1000);
  slider_layout->addWidget(slider);

  end_time_label = new QLabel(formatTime(can->totalSeconds()));
  slider_layout->addWidget(end_time_label);
  main_layout->addLayout(slider_layout);

  // btn controls
  QHBoxLayout *control_layout = new QHBoxLayout();
  play_btn = new QPushButton("⏸");
  play_btn->setStyleSheet("font-weight:bold");
  control_layout->addWidget(play_btn);

  QButtonGroup *group = new QButtonGroup(this);
  group->setExclusive(true);
  for (float speed : {0.1, 0.5, 1., 2.}) {
    QPushButton *btn = new QPushButton(QString("%1x").arg(speed), this);
    btn->setCheckable(true);
    QObject::connect(btn, &QPushButton::clicked, [=]() { can->setSpeed(speed); });
    control_layout->addWidget(btn);
    group->addButton(btn);
    if (speed == 1.0) btn->setChecked(true);
  }
  main_layout->addLayout(control_layout);

  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

  QObject::connect(can, &CANMessages::rangeChanged, this, &VideoWidget::rangeChanged);
  QObject::connect(can, &CANMessages::updated, this, &VideoWidget::updateState);
  QObject::connect(slider, &QSlider::sliderReleased, [this]() { can->seekTo(slider->value() / 1000.0); });
  QObject::connect(slider, &QSlider::valueChanged, [=](int value) { time_label->setText(formatTime(value / 1000)); });
  QObject::connect(cam_widget, &CameraViewWidget::clicked, [this]() { pause(!can->isPaused()); });
  QObject::connect(play_btn, &QPushButton::clicked, [=]() { pause(!can->isPaused()); });
}

void VideoWidget::pause(bool pause) {
  play_btn->setText(!pause ? "⏸" : "▶");
  can->pause(pause);
}

void VideoWidget::rangeChanged(double min, double max) {
  if (!can->isZoomed()) {
    min = 0;
    max = can->totalSeconds();
  }
  end_time_label->setText(formatTime(max));
  slider->setMinimum(min * 1000);
  slider->setMaximum(max * 1000);
}

void VideoWidget::updateState() {
  if (!slider->isSliderDown())
    slider->setValue(can->currentSec() * 1000);
}

// Slider
Slider::Slider(QWidget *parent) : QSlider(Qt::Horizontal, parent) {
  QTimer *timer = new QTimer(this);
  timer->setInterval(2000);
  timer->callOnTimeout([this]() {
    timeline = can->getTimeline();
    update();
  });
  timer->start();
}

void Slider::sliderChange(QAbstractSlider::SliderChange change) {
  if (change == QAbstractSlider::SliderValueChange) {
    int x = width() * ((value() - minimum()) / double(maximum() - minimum()));
    if (x != slider_x) {
      slider_x = x;
      update();
    }
  } else {
    QAbstractSlider::sliderChange(change);
  }
}

void Slider::paintEvent(QPaintEvent *ev) {
  static const QColor colors[] = {
    [(int)TimelineType::None] = QColor(111, 143, 175),
    [(int)TimelineType::Engaged] = QColor(0, 163, 108),
    [(int)TimelineType::UserFlag] = Qt::white,
    [(int)TimelineType::AlertInfo] = Qt::green,
    [(int)TimelineType::AlertWarning] = QColor(255, 195, 0),
    [(int)TimelineType::AlertCritical] = QColor(199, 0, 57)};

  QPainter p(this);
  QRect r = rect().adjusted(0, 4, 0, -4);
  p.fillRect(r, colors[(int)TimelineType::None]);
  double min = minimum() / 1000.0;
  double max = maximum() / 1000.0;
  for (auto [begin, end, type] : timeline) {
    if (begin > max || end < min)
      continue;
    r.setLeft(((std::max(min, (double)begin) - min) / (max - min)) * width());
    r.setRight(((std::min(max, (double)end) - min) / (max - min)) * width());
    p.fillRect(r, colors[(int)type]);
  }

  QStyleOptionSlider opt;
  opt.initFrom(this);
  opt.minimum = minimum();
  opt.maximum = maximum();
  opt.subControls = QStyle::SC_SliderHandle;
  opt.sliderPosition = value();
  style()->drawComplexControl(QStyle::CC_Slider, &opt, &p);
}

void Slider::mousePressEvent(QMouseEvent *e) {
  QSlider::mousePressEvent(e);
  if (e->button() == Qt::LeftButton && !isSliderDown()) {
    int value = minimum() + ((maximum() - minimum()) * e->x()) / width();
    setValue(value);
    emit sliderReleased();
  }
}
