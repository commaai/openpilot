#include "tools/cabana/videowidget.h"

#include <QButtonGroup>
#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include "tools/cabana/parser.h"

inline QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh::mm::ss" : "mm::ss");
}

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // TODO: figure out why the CameraViewWidget crashed occasionally.
  cam_widget = new CameraViewWidget("camerad", VISION_STREAM_ROAD, false, this);
  cam_widget->setFixedSize(parent->width(), parent->width() / 1.596);
  main_layout->addWidget(cam_widget);

  // slider controls
  QHBoxLayout *slider_layout = new QHBoxLayout();
  time_label = new QLabel("00:00");
  slider_layout->addWidget(time_label);

  slider = new Slider(this);
  slider->setSingleStep(0);
  slider->setMinimum(0);
  slider->setMaximum(parser->replay->totalSeconds() * 1000);
  slider_layout->addWidget(slider);

  total_time_label = new QLabel(formatTime(parser->replay->totalSeconds()));
  slider_layout->addWidget(total_time_label);

  main_layout->addLayout(slider_layout);

  // btn controls
  QHBoxLayout *control_layout = new QHBoxLayout();
  QPushButton *play = new QPushButton("⏸");
  play->setStyleSheet("font-weight:bold");
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
  QObject::connect(slider, &QSlider::sliderMoved, [=]() { time_label->setText(formatTime(slider->value() / 1000)); });
  QObject::connect(slider, &QSlider::sliderReleased, [this]() { setPosition(slider->value()); });
  QObject::connect(slider, &Slider::setPosition, this, &VideoWidget::setPosition);

  QObject::connect(play, &QPushButton::clicked, [=]() {
    bool is_paused = parser->replay->isPaused();
    play->setText(is_paused ? "⏸" : "▶");
    parser->replay->pause(!is_paused);
  });
}

void VideoWidget::setPosition(int value) {
  time_label->setText(formatTime(value / 1000.0));
  parser->seekTo(value / 1000.0);
}

void VideoWidget::rangeChanged(double min, double max) {
  if (!parser->isZoomed()) {
    min = 0;
    max = parser->replay->totalSeconds();
  }
  time_label->setText(formatTime(min));
  total_time_label->setText(formatTime(max));
  slider->setMinimum(min * 1000);
  slider->setMaximum(max * 1000);
  slider->setValue(parser->currentSec() * 1000);
}

void VideoWidget::updateState() {
  if (!slider->isSliderDown()) {
    double current_sec = parser->currentSec();
    time_label->setText(formatTime(current_sec));
    slider->setValue(current_sec * 1000);
  }
}

// Slider
Slider::Slider(QWidget *parent) : QSlider(Qt::Horizontal, parent) {
  QTimer *timer = new QTimer(this);
  timer->setInterval(2000);
  timer->callOnTimeout([this]() {
    timeline = parser->replay->getTimeline();
    update();
  });
  timer->start();
}

void Slider::sliderChange(QAbstractSlider::SliderChange change) {
  if (change == QAbstractSlider::SliderValueChange) {
    qreal x = width() * ((value() - minimum()) / double(maximum() - minimum()));
    if (x != slider_x) {
      slider_x = x;
      update();
    }
  } else {
    QAbstractSlider::sliderChange(change);
  }
}

void Slider::paintEvent(QPaintEvent *ev) {
  auto getPaintRange = [this](double begin, double end) -> std::pair<double, double> {
    double total_sec = maximum() - minimum();
    int start_pos = ((std::max((double)minimum(), (double)begin) - minimum()) / total_sec) * width();
    int end_pos = ((std::min((double)maximum(), (double)end) - minimum()) / total_sec) * width();
    return {start_pos, end_pos};
  };

  QPainter p(this);
  const int v_margin = 2;
  p.fillRect(rect().adjusted(0, v_margin, 0, -v_margin), QColor(111, 143, 175));

  for (auto [begin, end, type] : timeline) {
    begin *= 1000;
    end *= 1000;
    if (begin > maximum() || end < minimum()) continue;

    if (type == TimelineType::Engaged) {
      auto [start_pos, end_pos] = getPaintRange(begin, end);
      p.fillRect(QRect(start_pos, v_margin, end_pos - start_pos, height() - v_margin * 2), QColor(0, 163, 108));
    }
  }
  for (auto [begin, end, type] : timeline) {
    begin *= 1000;
    end *= 1000;
    if (type == TimelineType::Engaged || begin > maximum() || end < minimum()) continue;

    auto [start_pos, end_pos] = getPaintRange(begin, end);
    if (type == TimelineType::UserFlag) {
      p.fillRect(QRect(start_pos, height() - v_margin - 3, end_pos - start_pos, 3), Qt::white);
    } else {
      QColor color(Qt::green);
      if (type != TimelineType::AlertInfo)
        color = type == TimelineType::AlertWarning ? QColor(255, 195, 0) : QColor(199, 0, 57);

      p.fillRect(QRect(start_pos, height() - v_margin - 3, end_pos - start_pos, 3), color);
    }
  }
  p.setPen(QPen(QColor(88, 24, 69), 3));
  p.drawLine(QPoint{slider_x, 0}, QPoint{slider_x, height()});
}

void Slider::mousePressEvent(QMouseEvent *e) {
  QSlider::mousePressEvent(e);
  if (e->button() == Qt::LeftButton && !isSliderDown()) {
    int value = minimum() + ((maximum() - minimum()) * e->x()) / width();
    setValue(value);
    emit setPosition(value);
  }
}
