#include "tools/cabana/videowidget.h"

#include <QButtonGroup>
#include <QDateTime>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionSlider>
#include <QVBoxLayout>
#include <QtConcurrent>

const int MIN_VIDEO_HEIGHT = 100;
const int THUMBNAIL_MARGIN = 3;

static const QColor timeline_colors[] = {
  [(int)TimelineType::None] = QColor(111, 143, 175),
  [(int)TimelineType::Engaged] = QColor(0, 163, 108),
  [(int)TimelineType::UserFlag] = Qt::magenta,
  [(int)TimelineType::AlertInfo] = Qt::green,
  [(int)TimelineType::AlertWarning] = QColor(255, 195, 0),
  [(int)TimelineType::AlertCritical] = QColor(199, 0, 57),
};

static inline QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
}

VideoWidget::VideoWidget(QWidget *parent) : QFrame(parent) {
  setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
  auto main_layout = new QVBoxLayout(this);
  if (!can->liveStreaming()) {
    main_layout->addWidget(createCameraWidget());
  }

  // btn controls
  QHBoxLayout *control_layout = new QHBoxLayout();
  play_btn = new QPushButton();
  play_btn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
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
  main_layout->addLayout(control_layout);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

  QObject::connect(play_btn, &QPushButton::clicked, []() { can->pause(!can->isPaused()); });
  QObject::connect(can, &AbstractStream::paused, this, &VideoWidget::updatePlayBtnState);
  QObject::connect(can, &AbstractStream::resume, this, &VideoWidget::updatePlayBtnState);
  updatePlayBtnState();

  setWhatsThis(tr(R"(
    <b>Video</b><br />
    <!-- TODO: add descprition here -->
    <span style="color:gray">Timeline color</span>
    <table>
    <tr><td><span style="color:%1;">■ </span>Disengaged </td>
        <td><span style="color:%2;">■ </span>Engaged</td></tr>
    <tr><td><span style="color:%3;">■ </span>User Flag </td>
        <td><span style="color:%4;">■ </span>Info</td></tr>
    <tr><td><span style="color:%5;">■ </span>Warning </td>
        <td><span style="color:%6;">■ </span>Critical</td></tr>
    </table>
    <span style="color:gray">Shortcuts</span><br/>
    Pause/Resume: <span style="background-color:lightGray;color:gray">&nbsp;space&nbsp;</span>
  )").arg(timeline_colors[(int)TimelineType::None].name(),
          timeline_colors[(int)TimelineType::Engaged].name(),
          timeline_colors[(int)TimelineType::UserFlag].name(),
          timeline_colors[(int)TimelineType::AlertInfo].name(),
          timeline_colors[(int)TimelineType::AlertWarning].name(),
          timeline_colors[(int)TimelineType::AlertCritical].name()));
}

QWidget *VideoWidget::createCameraWidget() {
  QWidget *w = new QWidget(this);
  QVBoxLayout *l = new QVBoxLayout(w);
  l->setContentsMargins(0, 0, 0, 0);
  cam_widget = new CameraWidget("camerad", can->visionStreamType(), false);
  cam_widget->setMinimumHeight(MIN_VIDEO_HEIGHT);
  cam_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  l->addWidget(cam_widget);

  // slider controls
  slider_layout = new QHBoxLayout();
  time_label = new QLabel("00:00");
  slider_layout->addWidget(time_label);

  slider = new Slider(this);
  slider->setSingleStep(0);
  slider_layout->addWidget(slider);

  end_time_label = new QLabel(this);
  slider_layout->addWidget(end_time_label);
  l->addLayout(slider_layout);
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

void VideoWidget::updatePlayBtnState() {
  play_btn->setIcon(utils::icon(can->isPaused() ? "play" : "pause"));
  play_btn->setToolTip(can->isPaused() ? tr("Play") : tr("Pause"));
}

// Slider
Slider::Slider(QWidget *parent) : timer(this), thumbnail_label(this), QSlider(Qt::Horizontal, parent) {
  timer.callOnTimeout([this]() {
    timeline = can->getTimeline();
    update();
  });
  setMouseTracking(true);
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
  timeline.clear();
  timer.start(2000);
  thumnail_future = QtConcurrent::run(this, &Slider::loadThumbnails);
}

void Slider::loadThumbnails() {
  const auto &segments = can->route()->segments();
  for (auto it = segments.rbegin(); it != segments.rend() && !abort_load_thumbnail; ++it) {
    std::string qlog = it->second.qlog.toStdString();
    if (!qlog.empty()) {
      LogReader log;
      if (log.load(qlog, &abort_load_thumbnail, {cereal::Event::Which::THUMBNAIL}, true, 0, 3)) {
        for (auto ev = log.events.cbegin(); ev != log.events.cend() && !abort_load_thumbnail; ++ev) {
          auto thumb = (*ev)->event.getThumbnail();
          auto data = thumb.getThumbnail();
          if (QPixmap pm; pm.loadFromData(data.begin(), data.size(), "jpeg")) {
            pm = pm.scaledToHeight(MIN_VIDEO_HEIGHT - THUMBNAIL_MARGIN * 2, Qt::SmoothTransformation);
            std::lock_guard lk(thumbnail_lock);
            thumbnails[thumb.getTimestampEof()] = pm;
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
  QPainter p(this);
  QRect r = rect().adjusted(0, 4, 0, -4);
  p.fillRect(r, timeline_colors[(int)TimelineType::None]);
  double min = minimum() / 1000.0;
  double max = maximum() / 1000.0;
  for (auto [begin, end, type] : timeline) {
    if (begin > max || end < min)
      continue;
    r.setLeft(((std::max(min, (double)begin) - min) / (max - min)) * width());
    r.setRight(((std::min(max, (double)end) - min) / (max - min)) * width());
    p.fillRect(r, timeline_colors[(int)type]);
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
  QPixmap thumb;
  double seconds = (minimum() + e->pos().x() * ((maximum() - minimum()) / (double)width())) / 1000.0;
  {
    std::lock_guard lk(thumbnail_lock);
    auto it = thumbnails.lowerBound((seconds + can->routeStartTime()) * 1e9);
    if (it != thumbnails.end()) thumb = it.value();
  }
  int x = std::clamp(e->pos().x() - thumb.width() / 2, THUMBNAIL_MARGIN, rect().right() - thumb.width() - THUMBNAIL_MARGIN);
  int y = -thumb.height() - THUMBNAIL_MARGIN - style()->pixelMetric(QStyle::PM_LayoutVerticalSpacing);
  thumbnail_label.showPixmap(mapToGlobal({x, y}), formatTime(seconds), thumb);
  QSlider::mouseMoveEvent(e);
}

void Slider::leaveEvent(QEvent *event) {
  thumbnail_label.hide();
  QSlider::leaveEvent(event);
}

// ThumbnailLabel

ThumbnailLabel::ThumbnailLabel(QWidget *parent) : QWidget(parent, Qt::Tool | Qt::FramelessWindowHint) {
  setAttribute(Qt::WA_ShowWithoutActivating);
  setVisible(false);
}

void ThumbnailLabel::showPixmap(const QPoint &pt, const QString &sec, const QPixmap &pm) {
  pixmap = pm;
  second = sec;
  setVisible(!pm.isNull());
  if (isVisible()) {
    setGeometry({pt, pm.size()});
    update();
  }
}

void ThumbnailLabel::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.drawPixmap(0, 0, pixmap);
  p.setPen(QPen(Qt::white, 2));
  p.drawRect(rect());
  p.drawText(rect().adjusted(0, 0, 0, -THUMBNAIL_MARGIN), second, Qt::AlignHCenter | Qt::AlignBottom);
}
