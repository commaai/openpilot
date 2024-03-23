#include "selfdrive/ui/qt/offroad/replay_controls.h"

#include <QDir>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <algorithm>
#include <utility>

#include "selfdrive/ui/ui.h"

static QString formatTime(int seconds) {
  return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
}

static std::map<QString, RoutesPanel::RouteItem> getRouteList() {
  std::map<QString, RoutesPanel::RouteItem> results;
  QDir log_dir(Path::log_root().c_str());
  for (const auto &folder : log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort)) {
    if (int pos = folder.lastIndexOf("--"); pos != -1) {
      if (QString route = folder.left(pos); !route.isEmpty()) {
        auto it = results.find(route);
        if (it == results.end()) {
          // check if segment is valid
          QString segment_path = log_dir.filePath(folder);
          auto segment_files = QDir(segment_path).entryList(QDir::Files);
          if (std::count_if(segment_files.cbegin(), segment_files.cend(),
                            [](auto &f) { return f == "rlog.bz2" || f == "qcamera.ts" || f == "rlog"; }) >= 2) {
            results[route] = RoutesPanel::RouteItem{
                .datetime = QFileInfo(segment_path).lastModified().toString(Qt::ISODate),
                .seconds = 60};
          }
        } else {
          uint64_t secs = (folder.right(pos - 2).toInt() + 1) * 60;
          it->second.seconds = std::max(secs, it->second.seconds);
        }
      }
    }
  }
  return results;
}

RoutesPanel::RoutesPanel(SettingsWindow *parent) : settings_window(parent), QWidget(parent) {
  main_layout = new QStackedLayout(this);
  main_layout->addWidget(not_available_label = new QLabel(tr("not available while onroad"), this));
  main_layout->addWidget(route_list_widget = new ListWidget(this));
  QObject::connect(uiState(), &UIState::offroadTransition, [this](bool offroad) {
    main_layout->setCurrentIndex(!offroad && !uiState()->replaying ? 0 : 1);
    need_refresh = true;
  });
}

void RoutesPanel::showEvent(QShowEvent *event) {
  if (uiState()->scene.started && !uiState()->replaying) {
    main_layout->setCurrentWidget(not_available_label);
    return;
  }
  main_layout->setCurrentWidget(route_list_widget);

  // async get route list
  if (need_refresh) {
    need_refresh = false;
    auto watcher = new QFutureWatcher<std::map<QString, RoutesPanel::RouteItem>>(this);
    QObject::connect(watcher, &QFutureWatcher<std::map<QString, RoutesPanel::RouteItem>>::finished, [=]() {
      updateRoutes(watcher->future().result());
      watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(getRouteList));
  }
}

void RoutesPanel::updateRoutes(const std::map<QString, RoutesPanel::RouteItem> &route_items) {
  // display last 100 routes
  // TODO: 1.display thumbnail. 2.display all local routes. 3.display remote routes. 4.search routes by datetime
  int n = 0;
  for (auto it = route_items.crbegin(); it != route_items.crend() && n < 100; ++it, ++n) {
    ButtonControl *r = nullptr;
    if (n >= routes.size()) {
      r = routes.emplace_back(new ButtonControl(it->second.datetime, tr("REPLAY")));
      route_list_widget->addItem(r);
      QObject::connect(r, &ButtonControl::clicked, [this, r]() {
        emit settings_window->closeSettings();
        emit uiState()->startReplay(r->property("route").toString(), QString::fromStdString(Path::log_root()));
      });
    } else {
      r = routes[n];
    }
    r->setTitle(it->second.datetime);
    r->setValue(formatTime(it->second.seconds));
    r->setProperty("route", it->first);
    r->setVisible(true);
  }
  for (; n < routes.size(); ++n) {
    routes[n]->setVisible(false);
  }
}

// class ReplayControls

ReplayControls::ReplayControls(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(UI_BORDER_SIZE, UI_BORDER_SIZE, UI_BORDER_SIZE, UI_BORDER_SIZE);

  controls_container = new QWidget(this);
  controls_container->setVisible(false);
  QHBoxLayout *controls_layout = new QHBoxLayout(controls_container);
  main_layout->addStretch(1);
  main_layout->addWidget(controls_container);

  QLabel *time_label = new QLabel(this);
  controls_layout->addWidget(time_label);
  controls_layout->addWidget(play_btn = new QPushButton(tr("PAUSE"), this));
  controls_layout->addWidget(slider = new QSlider(Qt::Horizontal, this));
  controls_layout->addWidget(stop_btn = new QPushButton(tr("STOP"), this));
  controls_layout->addWidget(end_time_label = new QLabel(this));

  slider->setSingleStep(0);
  slider->setPageStep(0);
  setStyleSheet(R"(
    * {font-size: 35px;font-weight:500;color:white}
    QPushButton {padding: 30px;border-radius: 20px;color: #E4E4E4;background-color: #393939;}
    QSlider {height: 68px;}
    QSlider::groove:horizontal {border: 1px solid #262626; height: 20px;background: #393939;}
    QSlider::sub-page {background: #33Ab4C;}
    QSlider::handle:horizontal {background: white;border-radius: 30px;width: 60px;height: 60px;margin: -20px 0px;}
  )");

  QObject::connect(slider, &QSlider::sliderReleased, [this]() { replay->seekTo(slider->sliderPosition(), false); });
  QObject::connect(slider, &QSlider::valueChanged, [=](int value) { time_label->setText(formatTime(value)); });
  QObject::connect(stop_btn, &QPushButton::clicked, this, &ReplayControls::stop);
  QObject::connect(play_btn, &QPushButton::clicked, [this]() {
    replay->pause(!replay->isPaused());
    play_btn->setText(replay->isPaused() ? tr("PLAY") : tr("PAUSE"));
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
