#include "tools/cabana/videowidget.h"

#include <QBuffer>
#include <QButtonGroup>
#include <QDateTime>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QStyleOptionSlider>
#include <QTimeEdit>
#include <QTimer>
#include <QToolTip>
#include <QVBoxLayout>
#include <QtConcurrent>

inline QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
}

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QFrame *frame = new QFrame(this);
  frame->setFrameShape(QFrame::StyledPanel);
  frame->setFrameShadow(QFrame::Sunken);
  main_layout->addWidget(frame);

  QVBoxLayout *frame_layout = new QVBoxLayout(frame);
  if (!can->liveStreaming()) {
    frame_layout->addWidget(createCameraWidget());
  }

  // btn controls
  QHBoxLayout *control_layout = new QHBoxLayout();
  play_btn = new QPushButton();
  control_layout->addWidget(play_btn);

  QButtonGroup *group = new QButtonGroup(this);
  group->setExclusive(true);
  for (float speed : {0.1, 0.5, 1., 2.}) {
    if (can->liveStreaming() && speed > 1) continue;

    QPushButton *btn = new QPushButton(QString("%1x").arg(speed), this);
    btn->setCheckable(true);
    QObject::connect(btn, &QPushButton::clicked, [=]() { can->setSpeed(speed); });
    control_layout->addWidget(btn);
    group->addButton(btn);
    if (speed == 1.0) btn->setChecked(true);
  }
  frame_layout->addLayout(control_layout);

  QObject::connect(play_btn, &QPushButton::clicked, []() { can->pause(!can->isPaused()); });
  QObject::connect(can, &AbstractStream::paused, this, &VideoWidget::updatePlayBtnState);
  QObject::connect(can, &AbstractStream::resume, this, &VideoWidget::updatePlayBtnState);
  updatePlayBtnState();
}

QWidget *VideoWidget::createCameraWidget() {
  QWidget *w = new QWidget(this);
  QVBoxLayout *l = new QVBoxLayout(w);
  l->setContentsMargins(0, 0, 0, 0);
  cam_widget = new CameraWidget("camerad", can->visionStreamType(), false);
  l->addWidget(cam_widget);

  // slider controls
  slider_layout = new QHBoxLayout();
  time_label = new ElidedLabel("00:00");
  time_label->setToolTip(tr("Click to set current time"));
  slider_layout->addWidget(time_label);

  slider = new Slider(this);
  slider->setSingleStep(0);
  slider_layout->addWidget(slider);

  end_time_label = new QLabel(this);
  slider_layout->addWidget(end_time_label);
  l->addLayout(slider_layout);

  cam_widget->setMinimumHeight(100);
  cam_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

  QObject::connect(time_label, &ElidedLabel::clicked, this, &VideoWidget::timeLabelClicked);
  QObject::connect(slider, &QSlider::sliderReleased, [this]() { can->seekTo(slider->value() / 1000.0); });
  QObject::connect(slider, &QSlider::valueChanged, [=](int value) { time_label->setText(formatTime(value / 1000)); });
  QObject::connect(cam_widget, &CameraWidget::clicked, []() { can->pause(!can->isPaused()); });
  QObject::connect(can, &AbstractStream::updated, this, &VideoWidget::updateState);
  QObject::connect(can, &AbstractStream::streamStarted, [this]() {
    end_time_label->setText(formatTime(can->totalSeconds()));
    slider->setRange(0, can->totalSeconds() * 1000);
  });
  return w;
}

void VideoWidget::timeLabelClicked() {
  auto time_edit = new QTimeEdit(this);
  auto init_date_time = can->currentDateTime();
  time_edit->setDateTime(init_date_time);
  time_edit->setDisplayFormat("hh:mm:ss");
  time_label->setVisible(false);
  slider_layout->insertWidget(0, time_edit);
  QTimer::singleShot(0, [=]() { time_edit->setFocus(); });

  QObject::connect(time_edit, &QTimeEdit::editingFinished, [=]() {
    if (time_edit->dateTime() != init_date_time) {
      int seconds = can->route()->datetime().secsTo(time_edit->dateTime());
      can->seekTo(seconds);
    }
    time_edit->setVisible(false);
    time_label->setVisible(true);
    time_edit->deleteLater();
  });
}

void VideoWidget::rangeChanged(double min, double max, bool is_zoomed) {
  if (can->liveStreaming()) return;

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

void VideoWidget::updatePlayBtnState() {
  play_btn->setIcon(utils::icon(can->isPaused() ? "play" : "pause"));
  play_btn->setToolTip(can->isPaused() ? tr("Play") : tr("Pause"));
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
  QObject::connect(can, &AbstractStream::streamStarted, this, &Slider::streamStarted);
}

Slider::~Slider() {
  abort_load_thumbnail = true;
  thumnail_future.waitForFinished();
}

void Slider::streamStarted() {
  abort_load_thumbnail = true;
  thumnail_future.waitForFinished();
  abort_load_thumbnail = false;
  thumbnails.clear();
  thumnail_future = QtConcurrent::run(this, &Slider::loadThumbnails);
}

void Slider::loadThumbnails() {
  const auto segments = can->route()->segments();
  for (auto it = segments.rbegin(); it != segments.rend() && !abort_load_thumbnail; ++it) {
    std::string qlog = it->second.qlog.toStdString();
    if (!qlog.empty()) {
      LogReader log;
      if (log.load(qlog, &abort_load_thumbnail, {cereal::Event::Which::THUMBNAIL}, true, 0, 3)) {
        for (auto ev = log.events.cbegin(); ev != log.events.cend() && !abort_load_thumbnail; ++ev) {
          auto thumb = (*ev)->event.getThumbnail();
          QString str = getThumbnailString(thumb.getThumbnail());
          std::lock_guard lk(thumbnail_lock);
          thumbnails[thumb.getTimestampEof()] = str;
        }
      }
    }
  }
}

QString Slider::getThumbnailString(const capnp::Data::Reader &data) {
  QPixmap thumb;
  if (thumb.loadFromData(data.begin(), data.size(), "jpeg")) {
    thumb = thumb.scaled({thumb.width()/3, thumb.height()/3}, Qt::KeepAspectRatio);
    thumbnail_size = thumb.size();
    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    thumb.save(&buffer, "png");
    return  QString("<img src='data:image/png;base64, %0'>").arg(QString(bytes.toBase64()));
  }
  return {};
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
    double seconds = (minimum() + e->pos().x() * ((maximum() - minimum()) / (double)width())) / 1000.0;
    std::lock_guard lk(thumbnail_lock);
    auto it = thumbnails.lowerBound((seconds + can->routeStartTime()) * 1e9);
    if (it != thumbnails.end()) {
      thumb = it.value();
    }
  }
  QPoint pt = mapToGlobal({e->pos().x() - thumbnail_size.width() / 2, -thumbnail_size.height() - 30});
  QToolTip::showText(pt, thumb, this, rect());
  QSlider::mouseMoveEvent(e);
}
