#include "selfdrive/ui/qt/offroad/replay_controls.h"

#include <QDir>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <algorithm>
#include <utility>

#include "selfdrive/ui/ui.h"

namespace {
// TODO: move to assert file
QString pause_svg = R"(<svg width="512" height="512" viewBox="0 0 512 512" style="color:#ffffff" xmlns="http://www.w3.org/2000/svg" class="h-full w-full"><rect width="512" height="512" x="0" y="0" rx="30" fill="transparent" stroke="transparent" stroke-width="0" stroke-opacity="100%" paint-order="stroke"></rect><svg width="125px" height="125px" viewBox="0 0 14 14" fill="#ffffff" x="193.5" y="193.5" role="img" style="display:inline-block;vertical-align:middle" xmlns="http://www.w3.org/2000/svg"><g fill="#ffffff"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="7" cy="7" r="6.5"/><path d="M5.5 4.5v5m3-5v5"/></g></g></svg></svg>)";
QString play_svg = R"(<svg width="512" height="512" viewBox="0 0 512 512" style="color:#ffffff" xmlns="http://www.w3.org/2000/svg" class="h-full w-full"><rect width="512" height="512" x="0" y="0" rx="30" fill="transparent" stroke="transparent" stroke-width="0" stroke-opacity="100%" paint-order="stroke"></rect><svg width="125px" height="125px" viewBox="0 0 14 14" fill="#ffffff" x="193.5" y="193.5" role="img" style="display:inline-block;vertical-align:middle" xmlns="http://www.w3.org/2000/svg"><g fill="#ffffff"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="7" cy="7" r="6.5"/><path d="m5.5 4.5l4 2.5l-4 2.5v-5z"/></g></g></svg></svg>)";
QString stop_svg = R"(<svg width="512" height="512" viewBox="0 0 512 512" style="color:#ffffff" xmlns="http://www.w3.org/2000/svg" class="h-full w-full"><rect width="512" height="512" x="0" y="0" rx="30" fill="transparent" stroke="transparent" stroke-width="0" stroke-opacity="100%" paint-order="stroke"></rect><svg width="125px" height="125px" viewBox="0 0 14 14" fill="#ffffff" x="193.5" y="193.5" role="img" style="display:inline-block;vertical-align:middle" xmlns="http://www.w3.org/2000/svg"><g fill="#ffffff"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="7" cy="7" r="6.5"/><rect width="5" height="5" x="4.5" y="4.5" rx="1"/></g></g></svg></svg>)";

const int THUMBNAIL_HEIGHT = 60;

QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
}

QString getThumbnailPath(const QString &route) {
  return QString::fromStdString(Path::comma_home()) + "/route_thumbnail/" + route + ".jpeg";
}

void setThumbnail(ButtonControl *btn, const QPixmap &thumbnail) {
  btn->icon_label->setPixmap(btn->icon_pixmap = thumbnail);
  btn->icon_label->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  btn->icon_label->setVisible(true);
}
}  // namespace

RoutesPanel::RoutesPanel(SettingsWindow *parent) : settings_window(parent), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addWidget(route_list_widget = new ListWidget(this));
  QObject::connect(this, &RoutesPanel::thumbnailReady, this, &RoutesPanel::updateThumbnail, Qt::QueuedConnection);
  QObject::connect(uiState(), &UIState::offroadTransition, [this](bool offroad) {
    if (offroad && !uiState()->replaying) {
      fetchRoutes();
    }
    for (auto &r : routes) {
      r->setEnabled(offroad || uiState()->replaying);
    }
  });
}

void RoutesPanel::showEvent(QShowEvent *event) {
  if (routes.empty() && !uiState()->scene.started) {
    fetchRoutes();
  }
}

// async get route list
void RoutesPanel::fetchRoutes() {
  auto watcher = new QFutureWatcher<std::map<QString, RoutesPanel::RouteItem>>(this);
  QObject::connect(watcher, &QFutureWatcher<std::map<QString, RoutesPanel::RouteItem>>::finished, [=]() {
    updateRouteItems(watcher->future().result());
    watcher->deleteLater();
  });
  watcher->setFuture(QtConcurrent::run(this, &RoutesPanel::getRouteList));
}

void RoutesPanel::updateRouteItems(const std::map<QString, RoutesPanel::RouteItem> &route_items) {
  // display last 100 routes
  // TODO: 1.display all local routes. 2.display remote routes. 3.search routes by datetime
  int n = 0;
  for (auto it = route_items.crbegin(); it != route_items.crend() && n < 100; ++it, ++n) {
    ButtonControl *r = nullptr;
    if (n >= routes.size()) {
      r = routes.emplace_back(new ButtonControl(it->second.datetime, tr("REPLAY")));
      route_list_widget->addItem(r);
      QObject::connect(r, &ButtonControl::clicked, [this, r]() {
        if (uiState()->replaying) {
          emit settings_window->closeSettings();
        }
        emit uiState()->startReplay(r->property("route").toString(), QString::fromStdString(Path::log_root()));
      });
    } else {
      r = routes[n];
    }
    r->setTitle(it->second.datetime);
    r->setValue(formatTime(it->second.seconds));
    r->setProperty("route", it->first);
    setThumbnail(r, QPixmap(getThumbnailPath(it->first)));
    r->setVisible(true);
  }
  for (; n < routes.size(); ++n) {
    routes[n]->setVisible(false);
  }
}

void RoutesPanel::updateThumbnail(const QString route) {
  auto it = std::find_if(routes.begin(), routes.end(), [&](auto &r) { return r->property("route").toString() == route; });
  if (it != routes.end()) {
    setThumbnail(*it, QPixmap(getThumbnailPath(route)));
  }
}

std::map<QString, RoutesPanel::RouteItem> RoutesPanel::getRouteList() {
  std::map<QString, RoutesPanel::RouteItem> results;
  QDir log_dir(Path::log_root().c_str());
  for (const auto &segment : log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort)) {
    if (int pos = segment.lastIndexOf("--"); pos != -1) {
      if (QString route = segment.left(pos); !route.isEmpty()) {
        if (auto it = results.find(route); it == results.end()) {
          QString segment_path = log_dir.filePath(segment);
          auto segment_files = QDir(segment_path).entryList(QDir::Files);
          // check if segment is valid
          if (std::count_if(segment_files.cbegin(), segment_files.cend(),
                            [](auto &f) { return f == "rlog.bz2" || f == "qcamera.ts" || f == "rlog"; }) >= 2) {
            results[route] = {.datetime = QFileInfo(segment_path).lastModified().toString(Qt::ISODate),
                              .seconds = 60};
            if (QPixmap thumbnail(getThumbnailPath(route)); thumbnail.isNull()) {
              QtConcurrent::run(this, &RoutesPanel::extractThumbnal, route, segment_path);
            }
          }
        } else {
          uint64_t secs = (segment.right(pos - 2).toInt() + 1) * 60;
          it->second.seconds = std::max(secs, it->second.seconds);
        }
      }
    }
  }
  return results;
}

void RoutesPanel::extractThumbnal(QString route_name, QString segment_path) {
  QString qlog_fn = segment_path + "/qlog";
  if (!QFileInfo::exists(qlog_fn)) {
    qlog_fn += ".bz2";
  }
  if (LogReader log; log.load(qlog_fn.toStdString())) {
    auto it = std::find_if(log.events.cbegin(), log.events.cend(), [](auto e) { return e->which == cereal::Event::Which::THUMBNAIL; });
    if (it != log.events.cend()) {
      auto thumb = (*it)->event.getThumbnail().getThumbnail();
      if (QPixmap pm; pm.loadFromData(thumb.begin(), thumb.size(), "jpeg")) {
        QString fn = getThumbnailPath(route_name);
        QDir().mkpath(QFileInfo(fn).absolutePath());
        pm.scaledToHeight(THUMBNAIL_HEIGHT, Qt::SmoothTransformation).save(fn);
        emit thumbnailReady(route_name);
      }
    }
  }
}

// class ReplayControls

ReplayControls::ReplayControls(QWidget *parent) : QWidget(parent) {
  // TODO: add more buttons (prev, next, etc.)
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(UI_BORDER_SIZE, UI_BORDER_SIZE, UI_BORDER_SIZE, UI_BORDER_SIZE);

  controls_container = new QWidget(this);
  controls_container->setVisible(false);
  QVBoxLayout *controls_layout = new QVBoxLayout(controls_container);
  controls_layout->setContentsMargins(192 + UI_BORDER_SIZE, 12, 192 + UI_BORDER_SIZE, 12);
  main_layout->addStretch(1);
  main_layout->addWidget(controls_container);

  QHBoxLayout *controls_top_layout = new QHBoxLayout();
  QHBoxLayout *controls_bottom_layout = new QHBoxLayout();

  QLabel *time_label = new QLabel(this);
  controls_top_layout->addWidget(time_label);
  controls_top_layout->addWidget(slider = new QSlider(Qt::Horizontal, this));
  controls_top_layout->addWidget(end_time_label = new QLabel(this));
  controls_bottom_layout->setSpacing(25);
  controls_bottom_layout->addStretch(1);
  controls_bottom_layout->addWidget(play_btn = new QPushButton(this));
  controls_bottom_layout->addWidget(stop_btn = new QPushButton(this));
  controls_bottom_layout->addStretch(1);

  const QSize icon_size(100, 100);
  play_icon = QPixmap::fromImage(QImage::fromData(play_svg.toUtf8())).scaled(icon_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  pause_icon = QPixmap::fromImage(QImage::fromData(pause_svg.toUtf8())).scaled(icon_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  play_btn->setIconSize(icon_size);
  play_btn->setIcon(pause_icon);
  stop_btn->setIconSize(icon_size);
  stop_btn->setIcon(QPixmap::fromImage(QImage::fromData(stop_svg.toUtf8())).scaled(icon_size, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  controls_layout->addLayout(controls_top_layout);
  controls_layout->addLayout(controls_bottom_layout);

  slider->setSingleStep(0);
  slider->setPageStep(0);
  setStyleSheet(R"(
    * {font-size: 35px;font-weight:500;color:white}
    QPushButton {border:none;background:transparent;}
    QSlider {height: 48px;}
    QSlider::groove:horizontal {border: 1px solid #262626; height: 4px;background: #393939;}
    QSlider::sub-page {background: #33Ab4C;}
    QSlider::handle:horizontal {background: white;border-radius: 20px;width: 40px;height: 40px;margin: -18px 0px;}
  )");

  QObject::connect(slider, &QSlider::sliderReleased, [this]() { replay->seekTo(slider->sliderPosition(), false); });
  QObject::connect(slider, &QSlider::valueChanged, [=](int value) { time_label->setText(formatTime(value)); });
  QObject::connect(stop_btn, &QPushButton::clicked, this, &ReplayControls::stop);
  QObject::connect(play_btn, &QPushButton::clicked, [this]() {
    replay->pause(!replay->isPaused());
    play_btn->setIcon(replay->isPaused() ? play_icon : pause_icon);
  });

  timer = new QTimer(this);
  timer->setInterval(1000);
  timer->callOnTimeout([this]() {
    if (!slider->isSliderDown()) {
      slider->setValue(replay->currentSeconds());
    }
  });
  timer->start();

  time_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  time_label->setText("99:99:99");
  time_label->setFixedSize(time_label->sizeHint());
  end_time_label->setFixedSize(time_label->sizeHint());
  time_label->setText("");
  setVisible(true);
  adjustPosition();
}

void ReplayControls::paintEvent(QPaintEvent *event) {
  if (!route_loaded) {
    QPainter p(this);
    p.setPen(Qt::white);
    p.setRenderHint(QPainter::TextAntialiasing);
    p.setFont(InterFont(100, QFont::Bold));
    p.fillRect(rect(), Qt::black);
    p.drawText(geometry(), Qt::AlignCenter, tr("loading route"));
  }
}

void ReplayControls::adjustPosition() {
  setGeometry(0, 0, parentWidget()->width(), parentWidget()->height());
  raise();
}

void ReplayControls::start(const QString &route, const QString &data_dir) {
  QStringList allow = {"modelV2", "controlsState", "liveCalibration", "radarState", "roadCameraState",
                       "roadEncodeIdx", "carParams", "driverMonitoringState", "carState", "liveLocationKalman",
                       "driverStateV2", "wideRoadCameraState", "navInstruction", "navRoute", "uiPlan"};
  QString route_name = "0000000000000000|" + route;
  replay.reset(new Replay(route_name, allow, {}, nullptr, REPLAY_FLAG_QCAMERA | REPLAY_FLAG_LESS_CPU_USAGE, data_dir));
  QObject::connect(replay.get(), &Replay::streamStarted, [this]() {
    route_loaded = true;
    controls_container->setVisible(true);
    update();
  });
  if (replay->load()) {
    slider->setRange(0, replay->totalSeconds());
    end_time_label->setText(formatTime(replay->totalSeconds()));
    replay->start();
  }
}

void ReplayControls::stop() {
  if (replay) {
    timer->stop();
    replay->stop();
    QTimer::singleShot(1, []() { emit uiState()->stopReplay(); });
  }
}
