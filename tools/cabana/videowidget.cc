#include "tools/cabana/videowidget.h"

#include <QBuffer>
#include <QButtonGroup>
#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionSlider>
#include <QTimer>
#include <QToolTip>
#include <QVBoxLayout>
#include <QtConcurrent>

inline QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
}

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  cam_widget = new CameraWidget("camerad", VISION_STREAM_ROAD, false, this);
  cam_widget->setFixedSize(parent->width(), parent->width() / 1.596);
  main_layout->addWidget(cam_widget);

  // slider controls
  QHBoxLayout *slider_layout = new QHBoxLayout();
  QLabel *time_label = new QLabel("00:00");
  slider_layout->addWidget(time_label);

  slider = new Slider(this);
  slider->setSingleStep(0);
  slider_layout->addWidget(slider);

  end_time_label = new QLabel(this);
  slider_layout->addWidget(end_time_label);
  main_layout->addLayout(slider_layout);

  // btn controls
  QHBoxLayout *control_layout = new QHBoxLayout();
  play_btn = new QPushButton("⏸");
  play_btn->setStyleSheet("font-weight:bold; height:16px");
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

  QObject::connect(can, &CANMessages::updated, this, &VideoWidget::updateState);
  QObject::connect(slider, &QSlider::sliderReleased, [this]() { can->seekTo(slider->value() / 1000.0); });
  QObject::connect(slider, &QSlider::valueChanged, [=](int value) { time_label->setText(formatTime(value / 1000)); });
  QObject::connect(cam_widget, &CameraWidget::clicked, [this]() { pause(!can->isPaused()); });
  QObject::connect(play_btn, &QPushButton::clicked, [=]() { pause(!can->isPaused()); });
  QObject::connect(can, &CANMessages::streamStarted, [this]() {
    end_time_label->setText(formatTime(can->totalSeconds()));
    slider->setRange(0, can->totalSeconds() * 1000);
  });
}

void VideoWidget::pause(bool pause) {
  play_btn->setText(!pause ? "⏸" : "▶");
  can->pause(pause);
}

void VideoWidget::rangeChanged(double min, double max, bool is_zoomed) {
  if (!is_zoomed) {
    min = 0;
    max = can->totalSeconds();
  }
  end_time_label->setText(formatTime(max));
  slider->setRange(min * 1000, max * 1000);
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
  setMouseTracking(true);

  QObject::connect(can, SIGNAL(streamStarted()), timer, SLOT(start()));
  QObject::connect(can, &CANMessages::streamStarted, [this]() {
    abort_load_thumbnail = true;
    thumnail_future.waitForFinished();
    abort_load_thumbnail = false;
    thumbnails.clear();
    thumnail_future = QtConcurrent::run(this, &Slider::loadThumbnails);
  });
}

void Slider::loadThumbnails() {
  const auto &segments = can->route()->segments();
  for (int i = 0; i < segments.size() && !abort_load_thumbnail; ++i) {
    std::string qlog = segments.at(i).qlog.toStdString();
    if (!qlog.empty()) {
      LogReader log;
      if (log.load(qlog, &abort_load_thumbnail, {cereal::Event::Which::THUMBNAIL}, true, 0, 3)) {
        for (auto it = log.events.cbegin(); it != log.events.cend() && !abort_load_thumbnail; ++it) {
          auto thumb = (*it)->event.getThumbnail();
          auto data = thumb.getThumbnail();
          QPixmap pic;
          if (pic.loadFromData(data.begin(), data.size(), "jpeg")) {
            pic = pic.scaled({pic.width()/3, pic.height()/3}, Qt::KeepAspectRatio);
            thumbnail_size = pic.size();
            QByteArray bytes;
            QBuffer buffer(&bytes);
            buffer.open(QIODevice::WriteOnly);
            pic.save(&buffer, "jpeg");
            std::lock_guard lk(lock);
            thumbnails[thumb.getTimestampEof()] = QString("<img src='data:image/jpeg;base64, %0'>").arg(QString(bytes.toBase64()));
          }
        }
      }
    }
  }
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

void Slider::mouseMoveEvent(QMouseEvent *e) {
  QString thumb;
  {
    uint64_t ts = ((e->pos().x() * ((maximum() / 1000.0 - minimum() / 1000.0) / (double)width())) + can->routeStartTime()) * 1e9;
    std::lock_guard lk(lock);
    auto it = thumbnails.lowerBound(ts);
    if (it != thumbnails.end()) {
      thumb = it.value();
    }
  }
  if (!thumb.isEmpty()) {
    QPoint pt = mapToGlobal({e->pos().x() - thumbnail_size.width() / 2, -thumbnail_size.height() - 26});
    QToolTip::showText(pt, thumb, this, QRect(), 5000);
  } else {
    QToolTip::hideText();
  }
  QSlider::mouseMoveEvent(e);
}

void Slider::leaveEvent(QEvent *event) {
  QToolTip::hideText();
  QSlider::leaveEvent(event);
}
