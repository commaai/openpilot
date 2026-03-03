#include "tools/cabana/videowidget.h"

#include <algorithm>
#include <cmath>
#include <thread>

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GLES3/gl3.h>
#endif

#include <QAction>
#include <QActionGroup>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionSlider>
#include <QVBoxLayout>

#include "imgui.h"
#include "tools/cabana/tools/routeinfo.h"

const int MIN_VIDEO_HEIGHT = 100;
const int THUMBNAIL_MARGIN = 3;

// Indexed by TimelineType: None, Engaged, AlertInfo, AlertWarning, AlertCritical, UserBookmark
static const QColor timeline_colors[] = {
  QColor(111, 143, 175),
  QColor(0, 163, 108),
  Qt::green,
  QColor(255, 195, 0),
  QColor(199, 0, 57),
  Qt::magenta,
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
          timeline_colors[(int)TimelineType::UserBookmark].name(),
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
  speed_btn->setMinimumWidth(speed_btn->fontMetrics().horizontalAdvance("0.05x  ") + style()->pixelMetric(QStyle::PM_MenuButtonIndicator));
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
  QRect handle_rect = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);
  QRect groove_rect = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);

  // Adjust groove height to match handle height
  int handle_height = handle_rect.height();
  groove_rect.setHeight(handle_height * 0.5);
  groove_rect.moveCenter(QPoint(groove_rect.center().x(), rect().center().y()));

  p.fillRect(groove_rect, timeline_colors[(int)TimelineType::None]);

  double min = minimum() / factor;
  double max = maximum() / factor;

  auto fillRange = [&](double begin, double end, const QColor &color) {
    if (begin > max || end < min) return;

    QRect r = groove_rect;
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
}

StreamCameraView::~StreamCameraView() {
  clearThumbnails();
}

void StreamCameraView::clearThumbnails() {
  makeCurrent();
  for (auto &[_, tex] : texture_cache) glDeleteTextures(1, &tex);
  doneCurrent();
  texture_cache.clear();
  thumbnails.clear();
  big_thumbnails.clear();
}

void StreamCameraView::parseQLog(std::shared_ptr<LogReader> qlog) {
  clearThumbnails();

  std::mutex mutex;
  const auto &events = qlog->events;
  unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
  size_t chunk = (events.size() + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < num_threads && t * chunk < events.size(); ++t) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, events.size());
    threads.emplace_back([this, &mutex, &events, start, end]() {
      for (size_t i = start; i < end; ++i) {
        const Event &e = events[i];
        if (e.which == cereal::Event::Which::THUMBNAIL) {
          capnp::FlatArrayMessageReader reader(e.data);
          auto thumb_data = reader.getRoot<cereal::Event>().getThumbnail();
          auto image_data = thumb_data.getThumbnail();
          QImage img;
          if (img.loadFromData(image_data.begin(), image_data.size(), "jpeg")) {
            QImage small = img.scaledToHeight(MIN_VIDEO_HEIGHT - THUMBNAIL_MARGIN * 2, Qt::SmoothTransformation);
            std::lock_guard lock(mutex);
            thumbnails[thumb_data.getTimestampEof()] = small.convertToFormat(QImage::Format_RGBA8888);
            big_thumbnails[thumb_data.getTimestampEof()] = img.convertToFormat(QImage::Format_RGBA8888);
          }
        }
      }
    });
  }
  for (auto &th : threads) th.join();
  update();
}

GLuint StreamCameraView::getOrUploadTexture(uint64_t key, const QImage &img) {
  auto it = texture_cache.find(key);
  if (it != texture_cache.end()) return it->second;

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.constBits());
  glBindTexture(GL_TEXTURE_2D, 0);

  texture_cache[key] = tex;
  return tex;
}

static void drawAlertOverlay(ImDrawList *draw, ImFont *font, const Timeline::Entry &alert, float x, float y, float w) {
  float font_size = font->LegacySize;
  const QColor &qc = timeline_colors[int(alert.type)];
  ImU32 bg_color = IM_COL32(qc.red(), qc.green(), qc.blue(), 128);
  std::string text = alert.text1;
  if (!alert.text2.empty()) text += "\n" + alert.text2;
  ImVec2 text_size = font->CalcTextSizeA(font_size, FLT_MAX, w, text.c_str());
  draw->AddRectFilled(ImVec2(x, y), ImVec2(x + w, y + text_size.y), bg_color);
  draw->AddText(font, font_size, ImVec2(x + (w - text_size.x) / 2.0f, y),
                IM_COL32(255, 255, 255, 255), text.c_str(), nullptr, w);
}

static void drawTimeLabel(ImDrawList *draw, ImFont *font, double seconds, float tx, float ty, float tw, float th) {
  float font_size = font->LegacySize;
  char time_buf[32];
  snprintf(time_buf, sizeof(time_buf), "%.3f", seconds);
  ImVec2 text_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, time_buf);
  draw->AddText(font, font_size,
                ImVec2(tx + (tw - text_size.x) / 2.0f, ty + th - text_size.y - THUMBNAIL_MARGIN),
                IM_COL32(255, 255, 255, 255), time_buf);
}

void StreamCameraView::drawImGuiOverlays() {
  auto *draw = ImGui::GetForegroundDrawList();
  float w = (float)width();
  float h = (float)height();
  ImFont *font_regular = imgui_font_regular ? imgui_font_regular : ImGui::GetFont();
  ImFont *font_bold = imgui_font_bold ? imgui_font_bold : ImGui::GetFont();
  Replay *replay = getReplay();
  bool paused = can->isPaused();
  bool scrubbing = false;

  // Thumbnails
  if (thumbnail_dispaly_time >= 0) {
    scrubbing = paused;
    if (scrubbing) {
      // Full-size scrub thumbnail
      auto it = big_thumbnails.lower_bound(can->toMonoTime(thumbnail_dispaly_time));
      if (it != big_thumbnails.end()) {
        const QImage &img = it->second;
        GLuint tex = getOrUploadTexture(it->first, img);

        float img_ratio = (float)img.width() / img.height();
        float widget_ratio = w / h;
        float tw, th;
        if (img_ratio > widget_ratio) {
          tw = w;
          th = w / img_ratio;
        } else {
          th = h;
          tw = h * img_ratio;
        }
        float tx = (w - tw) / 2.0f;
        float ty = (h - th) / 2.0f;

        draw->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), IM_COL32(0, 0, 0, 255));
        draw->AddImage((ImTextureID)(uintptr_t)tex, ImVec2(tx, ty), ImVec2(tx + tw, ty + th));
        drawTimeLabel(draw, font_regular, thumbnail_dispaly_time, tx, ty, tw, th);
      }
    } else {
      // Small hover thumbnail
      auto it = thumbnails.lower_bound(can->toMonoTime(thumbnail_dispaly_time));
      if (it != thumbnails.end()) {
        const QImage &img = it->second;
        GLuint tex = getOrUploadTexture(it->first | (1ULL << 63), img);  // separate cache key from big_thumbnails

        float tw = (float)img.width();
        float th = (float)img.height();
        auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
        float pos = (float)((thumbnail_dispaly_time - min_sec) * w / (max_sec - min_sec));
        float tx = std::clamp(pos - tw / 2.0f, (float)THUMBNAIL_MARGIN, w - tw - THUMBNAIL_MARGIN + 1.0f);
        float ty = h - th - THUMBNAIL_MARGIN;

        draw->AddRect(ImVec2(tx - 1, ty - 1), ImVec2(tx + tw + 1, ty + th + 1), IM_COL32(255, 255, 255, 255));
        draw->AddImage((ImTextureID)(uintptr_t)tex, ImVec2(tx, ty), ImVec2(tx + tw, ty + th));

        if (replay) {
          if (auto alert = replay->findAlertAtTime(thumbnail_dispaly_time))
            drawAlertOverlay(draw, font_regular, *alert, tx, ty, tw);
        }
        drawTimeLabel(draw, font_regular, thumbnail_dispaly_time, tx, ty, tw, th);
      }
    }
  }

  // Alert bar
  if (replay) {
    double alert_time = scrubbing ? thumbnail_dispaly_time : can->currentSec();
    if (auto alert = replay->findAlertAtTime(alert_time))
      drawAlertOverlay(draw, font_regular, *alert, 1.0f, 1.0f, w - 2.0f);
  }

  // PAUSED text
  if (paused) {
    const char *paused_text = "PAUSED";
    float font_size = font_bold->LegacySize;
    ImVec2 text_size = font_bold->CalcTextSizeA(font_size, FLT_MAX, 0.0f, paused_text);
    draw->AddText(font_bold, font_size,
                  ImVec2((w - text_size.x) / 2.0f, (h - text_size.y) / 2.0f),
                  IM_COL32(200, 200, 200, 128), paused_text);
  }
}
