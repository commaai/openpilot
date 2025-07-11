#include "tools/cabana/videowidget.h"

#include <algorithm>

#include <QAction>
#include <QActionGroup>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionSlider>
#include <QVBoxLayout>
#include <QtConcurrent>

#include "tools/cabana/tools/routeinfo.h"

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

static Replay *getReplay() {
  auto stream = qobject_cast<ReplayStream *>(can);
  return stream ? stream->getReplay() : nullptr;
}

VideoWidget::VideoWidget(QWidget *parent) : QFrame(parent) {
  setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
  auto main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  if (!can->liveStreaming())
    main_layout->addWidget(createCameraWidget());

  createPlaybackController();

  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  QObject::connect(can, &AbstractStream::paused, this, &VideoWidget::updatePlayBtnState);
  QObject::connect(can, &AbstractStream::resume, this, &VideoWidget::updatePlayBtnState);
  QObject::connect(can, &AbstractStream::msgsReceived, this, &VideoWidget::updateState);
  QObject::connect(can, &AbstractStream::seeking, this, &VideoWidget::updateState);
  QObject::connect(can, &AbstractStream::timeRangeChanged, this, &VideoWidget::timeRangeChanged);

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

void VideoWidget::createPlaybackController() {
  QToolBar *toolbar = new QToolBar(this);
  layout()->addWidget(toolbar);

  int icon_size = style()->pixelMetric(QStyle::PM_SmallIconSize);
  toolbar->setIconSize({icon_size, icon_size});

  toolbar->addAction(utils::icon("rewind"), tr("Seek backward"), []() { can->seekTo(can->currentSec() - 1); });
  play_toggle_action = toolbar->addAction(utils::icon("play"), tr("Play"), []() { can->pause(!can->isPaused()); });
  toolbar->addAction(utils::icon("fast-forward"), tr("Seek forward"), []() { can->seekTo(can->currentSec() + 1); });

  if (can->liveStreaming()) {
    skip_to_end_action = toolbar->addAction(utils::icon("skip-end"), tr("Skip to the end"), this, [this]() {
      // set speed to 1.0
      speed_btn->menu()->actions()[7]->setChecked(true);
      can->pause(false);
      can->seekTo(can->maxSeconds() + 1);
    });
  }

  time_display_action = toolbar->addAction("", this, [this]() {
    settings.absolute_time = !settings.absolute_time;
    time_display_action->setToolTip(settings.absolute_time ? tr("Elapsed time") : tr("Absolute time"));
    updateState();
  });

  QWidget *spacer = new QWidget();
  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(spacer);

  if (!can->liveStreaming()) {
    toolbar->addAction(utils::icon("repeat"), tr("Loop playback"), this, &VideoWidget::loopPlaybackClicked);
    createSpeedDropdown(toolbar);
    toolbar->addSeparator();
    toolbar->addAction(utils::icon("info-circle"), tr("View route details"), this, &VideoWidget::showRouteInfo);
  } else {
    createSpeedDropdown(toolbar);
  }
}

void VideoWidget::createSpeedDropdown(QToolBar *toolbar) {
  toolbar->addWidget(speed_btn = new QToolButton(this));
  speed_btn->setMenu(new QMenu(speed_btn));
  speed_btn->setPopupMode(QToolButton::InstantPopup);
  QActionGroup *speed_group = new QActionGroup(this);
  speed_group->setExclusive(true);

  for (float speed : {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1., 2., 3., 5.}) {
    auto act = speed_btn->menu()->addAction(QString("%1x").arg(speed), this, [this, speed]() {
      can->setSpeed(speed);
      speed_btn->setText(QString("%1x  ").arg(speed));
    });

    speed_group->addAction(act);
    act->setCheckable(true);
    if (speed == 1.0) {
      act->setChecked(true);
      act->trigger();
    }
  }

  QFont font = speed_btn->font();
  font.setBold(true);
  speed_btn->setFont(font);
  speed_btn->setMinimumWidth(speed_btn->fontMetrics().width("0.05x  ") + style()->pixelMetric(QStyle::PM_MenuButtonIndicator));
}

QWidget *VideoWidget::createCameraWidget() {
  QWidget *w = new QWidget(this);
  QVBoxLayout *l = new QVBoxLayout(w);
  l->setContentsMargins(0, 0, 0, 0);
  l->setSpacing(0);

  l->addWidget(camera_tab = new TabBar(w));
  camera_tab->setAutoHide(true);
  camera_tab->setExpanding(false);

  l->addWidget(cam_widget = new StreamCameraView("camerad", VISION_STREAM_ROAD));
  cam_widget->setMinimumHeight(MIN_VIDEO_HEIGHT);
  cam_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);

  l->addWidget(slider = new Slider(w));
  slider->setSingleStep(0);
  slider->setTimeRange(can->minSeconds(), can->maxSeconds());

  QObject::connect(slider, &QSlider::sliderReleased, [this]() { can->seekTo(slider->currentSecond()); });
  QObject::connect(can, &AbstractStream::paused, cam_widget, [c = cam_widget]() { c->showPausedOverlay(); });
  QObject::connect(can, &AbstractStream::eventsMerged, this, [this]() { slider->update(); });
  QObject::connect(cam_widget, &CameraWidget::clicked, []() { can->pause(!can->isPaused()); });
  QObject::connect(cam_widget, &CameraWidget::vipcAvailableStreamsUpdated, this, &VideoWidget::vipcAvailableStreamsUpdated);
  QObject::connect(camera_tab, &QTabBar::currentChanged, [this](int index) {
    if (index != -1) cam_widget->setStreamType((VisionStreamType)camera_tab->tabData(index).toInt());
  });
  QObject::connect(static_cast<ReplayStream*>(can), &ReplayStream::qLogLoaded, cam_widget, &StreamCameraView::parseQLog, Qt::QueuedConnection);
  slider->installEventFilter(this);
  return w;
}

void VideoWidget::vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams) {
  static const QString stream_names[] = {"Road camera", "Driver camera", "Wide road camera"};
  for (int i = 0; i < streams.size(); ++i) {
    if (camera_tab->count() <= i) {
      camera_tab->addTab(QString());
    }
    int type = *std::next(streams.begin(), i);
    camera_tab->setTabText(i, stream_names[type]);
    camera_tab->setTabData(i, type);
  }
  while (camera_tab->count() > streams.size()) {
    camera_tab->removeTab(camera_tab->count() - 1);
  }
}

void VideoWidget::loopPlaybackClicked() {
  bool is_looping = getReplay()->loop();
  getReplay()->setLoop(!is_looping);
  qobject_cast<QAction*>(sender())->setIcon(utils::icon(!is_looping ? "repeat" : "repeat-1"));
}

void VideoWidget::timeRangeChanged() {
  const auto time_range = can->timeRange();
  if (can->liveStreaming()) {
    skip_to_end_action->setEnabled(!time_range.has_value());
    return;
  }
  time_range ? slider->setTimeRange(time_range->first, time_range->second)
             : slider->setTimeRange(can->minSeconds(), can->maxSeconds());
  updateState();
}

QString VideoWidget::formatTime(double sec, bool include_milliseconds) {
  if (settings.absolute_time)
    sec = can->beginDateTime().addMSecs(sec * 1000).toMSecsSinceEpoch() / 1000.0;
  return utils::formatSeconds(sec, include_milliseconds, settings.absolute_time);
}

void VideoWidget::updateState() {
  if (slider) {
    if (!slider->isSliderDown()) {
      slider->setCurrentSecond(can->currentSec());
    }
    if (camera_tab->count() == 0) {  //  No streams available
      cam_widget->update();          // Manually refresh to show alert events
    }
    time_display_action->setText(QString("%1 / %2").arg(formatTime(can->currentSec(), true),
                                             formatTime(slider->maximum() / slider->factor)));
  } else {
    time_display_action->setText(formatTime(can->currentSec(), true));
  }
}

void VideoWidget::updatePlayBtnState() {
  play_toggle_action->setIcon(utils::icon(can->isPaused() ? "play" : "pause"));
  play_toggle_action->setToolTip(can->isPaused() ? tr("Play") : tr("Pause"));
}

void VideoWidget::showThumbnail(double seconds) {
  if (can->liveStreaming()) return;

  cam_widget->thumbnail_dispaly_time = seconds;
  slider->thumbnail_dispaly_time = seconds;
  cam_widget->update();
  slider->update();
}

void VideoWidget::showRouteInfo() {
  RouteInfoDlg *route_info = new RouteInfoDlg(this);
  route_info->setAttribute(Qt::WA_DeleteOnClose);
  route_info->show();
}

bool VideoWidget::eventFilter(QObject *obj, QEvent *event) {
  if (event->type() == QEvent::MouseMove) {
    auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
    showThumbnail(min_sec + static_cast<QMouseEvent *>(event)->pos().x() * (max_sec - min_sec) / slider->width());
  } else if (event->type() == QEvent::Leave) {
    showThumbnail(-1);
  }
  return false;
}

// Slider
Slider::Slider(QWidget *parent) : QSlider(Qt::Horizontal, parent) {
  setMouseTracking(true);
}

void Slider::paintEvent(QPaintEvent *ev) {
  QPainter p(this);

  QStyleOptionSlider opt;
  initStyleOption(&opt);
  QRect r = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
  p.fillRect(r, timeline_colors[(int)TimelineType::None]);

  double min = minimum() / factor;
  double max = maximum() / factor;

  auto fillRange = [&](double begin, double end, const QColor &color) {
    if (begin > max || end < min) return;
    r.setLeft(((std::max(min, begin) - min) / (max - min)) * width());
    r.setRight(((std::min(max, end) - min) / (max - min)) * width());
    p.fillRect(r, color);
  };

  if (auto replay = getReplay()) {
    for (const auto &entry : *replay->getTimeline()) {
      fillRange(entry.start_time, entry.end_time, timeline_colors[(int)entry.type]);
    }

    QColor empty_color = palette().color(QPalette::Window);
    empty_color.setAlpha(160);
    const auto event_data = replay->getEventData();
    for (const auto &[n, _] : replay->route().segments()) {
      if (!event_data->isSegmentLoaded(n))
        fillRange(n * 60.0, (n + 1) * 60.0, empty_color);
    }
  }

  opt.minimum = minimum();
  opt.maximum = maximum();
  opt.subControls = QStyle::SC_SliderHandle;
  opt.sliderPosition = value();
  style()->drawComplexControl(QStyle::CC_Slider, &opt, &p);

  if (thumbnail_dispaly_time >= 0) {
    int left = (thumbnail_dispaly_time - min) * width() / (max - min) - 1;
    QRect rc(left, rect().top() + 1, 2, rect().height() - 2);
    p.setBrush(palette().highlight());
    p.setPen(Qt::NoPen);
    p.drawRoundedRect(rc, 1.5, 1.5);
  }
}

void Slider::mousePressEvent(QMouseEvent *e) {
  QSlider::mousePressEvent(e);
  if (e->button() == Qt::LeftButton && !isSliderDown()) {
    setValue(minimum() + ((maximum() - minimum()) * e->x()) / width());
    emit sliderReleased();
  }
}

// StreamCameraView
StreamCameraView::StreamCameraView(std::string stream_name, VisionStreamType stream_type, QWidget *parent)
    : CameraWidget(stream_name, stream_type, parent) {
  fade_animation = new QPropertyAnimation(this, "overlayOpacity");
  fade_animation->setDuration(500);
  fade_animation->setStartValue(0.2f);
  fade_animation->setEndValue(0.7f);
  fade_animation->setEasingCurve(QEasingCurve::InOutQuad);
  connect(fade_animation, &QPropertyAnimation::valueChanged, this, QOverload<>::of(&StreamCameraView::update));
}

void StreamCameraView::parseQLog(std::shared_ptr<LogReader> qlog) {
  std::mutex mutex;
  QtConcurrent::blockingMap(qlog->events.cbegin(), qlog->events.cend(), [this, &mutex](const Event &e) {
    if (e.which == cereal::Event::Which::THUMBNAIL) {
      capnp::FlatArrayMessageReader reader(e.data);
      auto thumb_data = reader.getRoot<cereal::Event>().getThumbnail();
      auto image_data = thumb_data.getThumbnail();
      if (QPixmap thumb; thumb.loadFromData(image_data.begin(), image_data.size(), "jpeg")) {
        QPixmap generated_thumb = generateThumbnail(thumb, can->toSeconds(thumb_data.getTimestampEof()));
        std::lock_guard lock(mutex);
        thumbnails[thumb_data.getTimestampEof()] = generated_thumb;
        big_thumbnails[thumb_data.getTimestampEof()] = thumb;
      }
    }
  });
  update();
}

void StreamCameraView::paintGL() {
  CameraWidget::paintGL();

  QPainter p(this);
  bool scrubbing = false;
  if (thumbnail_dispaly_time >= 0) {
    scrubbing = can->isPaused();
    scrubbing ? drawScrubThumbnail(p) : drawThumbnail(p);
  }
  if (auto alert = getReplay()->findAlertAtTime(scrubbing ? thumbnail_dispaly_time : can->currentSec())) {
    drawAlert(p, rect(), *alert);
  }

  if (can->isPaused()) {
    p.setPen(QColor(200, 200, 200, static_cast<int>(255 * fade_animation->currentValue().toFloat())));
    p.setFont(QFont(font().family(), 16, QFont::Bold));
    p.drawText(rect(), Qt::AlignCenter, tr("PAUSED"));
  }
}

QPixmap StreamCameraView::generateThumbnail(QPixmap thumb, double seconds) {
  QPixmap scaled = thumb.scaledToHeight(MIN_VIDEO_HEIGHT - THUMBNAIL_MARGIN * 2, Qt::SmoothTransformation);
  QPainter p(&scaled);
  p.setPen(QPen(palette().color(QPalette::BrightText), 2));
  p.drawRect(scaled.rect());
  if (auto alert = getReplay()->findAlertAtTime(seconds)) {
    p.setFont(QFont(font().family(), 10));
    drawAlert(p, scaled.rect(), *alert);
  }
  return scaled;
}

void StreamCameraView::drawScrubThumbnail(QPainter &p) {
  p.fillRect(rect(), Qt::black);
  auto it = big_thumbnails.lowerBound(can->toMonoTime(thumbnail_dispaly_time));
  if (it != big_thumbnails.end()) {
    QPixmap scaled_thumb = it.value().scaled(rect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QRect thumb_rect(rect().center() - scaled_thumb.rect().center(), scaled_thumb.size());
    p.drawPixmap(thumb_rect.topLeft(), scaled_thumb);
    drawTime(p, thumb_rect, thumbnail_dispaly_time);
  }
}

void StreamCameraView::drawThumbnail(QPainter &p) {
  auto it = thumbnails.lowerBound(can->toMonoTime(thumbnail_dispaly_time));
  if (it != thumbnails.end()) {
    const QPixmap &thumb = it.value();
    auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
    int pos = (thumbnail_dispaly_time - min_sec) * width() / (max_sec - min_sec);
    int x = std::clamp(pos - thumb.width() / 2, THUMBNAIL_MARGIN, width() - thumb.width() - THUMBNAIL_MARGIN + 1);
    int y = height() - thumb.height() - THUMBNAIL_MARGIN;

    p.drawPixmap(x, y, thumb);
    drawTime(p, QRect{x, y, thumb.width(), thumb.height()}, thumbnail_dispaly_time);
  }
}

void StreamCameraView::drawTime(QPainter &p, const QRect &rect, double seconds) {
  p.setPen(palette().color(QPalette::BrightText));
  p.setFont(QFont(font().family(), 10));
  p.drawText(rect.adjusted(0, 0, 0, -THUMBNAIL_MARGIN), Qt::AlignHCenter | Qt::AlignBottom, QString::number(seconds, 'f', 3));
}

void StreamCameraView::drawAlert(QPainter &p, const QRect &rect, const Timeline::Entry &alert) {
  p.setPen(QPen(palette().color(QPalette::BrightText), 2));
  QColor color = timeline_colors[int(alert.type)];
  color.setAlphaF(0.5);
  QString text = QString::fromStdString(alert.text1);
  if (!alert.text2.empty()) text += "\n" + QString::fromStdString(alert.text2);

  QRect text_rect = rect.adjusted(1, 1, -1, -1);
  QRect r = p.fontMetrics().boundingRect(text_rect, Qt::AlignTop | Qt::AlignHCenter | Qt::TextWordWrap, text);
  p.fillRect(text_rect.left(), r.top(), text_rect.width(), r.height(), color);
  p.drawText(text_rect, Qt::AlignTop | Qt::AlignHCenter | Qt::TextWordWrap, text);
}
