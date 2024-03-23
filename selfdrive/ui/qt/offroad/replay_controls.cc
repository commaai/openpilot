#include "selfdrive/ui/qt/offroad/replay_controls.h"

#include <QDir>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <utility>

#include "selfdrive/ui/ui.h"

static std::map<QString, QString> getRouteList() {
  std::map<QString, QString> results;
  QDir log_dir(Path::log_root().c_str());
  for (const auto &folder : log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort)) {
    if (int pos = folder.lastIndexOf("--"); pos != -1) {
      if (QString route = folder.left(pos); !route.isEmpty() && results.count(route) == 0) {
        // check if segment is valid
        QString segment_path = log_dir.filePath(folder);
        auto segment_files = QDir(segment_path).entryList(QDir::Files);
        if (std::count_if(segment_files.cbegin(), segment_files.cend(),
                          [](auto &f) { return f == "rlog.bz2" || f == "qcamera.ts" || f == "rlog"; }) >= 2) {
          results[route] = QFileInfo(segment_path).lastModified().toString(Qt::ISODate);
        }
      }
    }
  }
  return results;
}

ReplayPanel::ReplayPanel(SettingsWindow *parent) : settings_window(parent), QWidget(parent) {
  main_layout = new QStackedLayout(this);
  main_layout->addWidget(not_available_label = new QLabel("not available while onroad", this));
  main_layout->addWidget(route_list_widget = new ListWidget(this));
  QObject::connect(uiState(), &UIState::offroadTransition, [this](bool offroad) {
    main_layout->setCurrentIndex(!offroad && !uiState()->replaying ? 0 : 1);
    need_refresh = true;
  });
}

void ReplayPanel::showEvent(QShowEvent *event) {
  if (uiState()->scene.started && !uiState()->replaying) {
    main_layout->setCurrentWidget(not_available_label);
    return;
  }
  main_layout->setCurrentWidget(route_list_widget);

  if (need_refresh) {
    need_refresh = false;
    auto watcher = new QFutureWatcher<std::map<QString, QString>>(this);
    QObject::connect(watcher, &QFutureWatcher<std::map<QString, QString>>::finished, [=]() {
      updateRoutes(watcher->future().result());
      watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(getRouteList));
  }
}

void ReplayPanel::updateRoutes(const std::map<QString, QString> &route_items) {
  // TODO: 1) display thumbnail, 2) feth all routes.
  int n = 0;
  for (auto it = route_items.crbegin(); it != route_items.crend() && n < 100; ++it, ++n) {
    ButtonControl *r = nullptr;
    if (n >= routes.size()) {
      r = routes.emplace_back(new ButtonControl(it->second, "REPLAY"));
      route_list_widget->addItem(r);
      QObject::connect(r, &ButtonControl::clicked, [this, r]() {
        emit settings_window->closeSettings();
        emit uiState()->startReplay(r->property("route").toString(), QString::fromStdString(Path::log_root()));
      });
    } else {
      r = routes[n];
    }
    r->setTitle(it->second);
    r->setProperty("route", it->first);
    r->setVisible(true);
  }
  for (; n < routes.size(); ++n) {
    routes[n]->setVisible(false);
  }
}

// class ReplayControls

ReplayControls::ReplayControls(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  QLabel *time_label = new QLabel("00:00");
  main_layout->addWidget(time_label);
  main_layout->addWidget(play_btn = new QPushButton("PAUSE", this));
  main_layout->addWidget(slider = new QSlider(Qt::Horizontal, this));
  main_layout->addWidget(stop_btn = new QPushButton("STOP", this));
  main_layout->addWidget(end_time_label = new QLabel(this));

  slider->setSingleStep(0);
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
    play_btn->setText(replay->isPaused() ? "PLAY" : "PAUSE");
  });

  timer = new QTimer(this);
  timer->setInterval(1000);
  timer->callOnTimeout([this]() {
    if (!slider->isSliderDown()) {
      slider->setValue(replay->currentSeconds());
    }
  });
  timer->start();
  setVisible(true);
  adjustPosition();
}

void ReplayControls::adjustPosition() {
  resize(parentWidget()->rect().width() - 100, sizeHint().height());
  move({50, parentWidget()->rect().height() - rect().height() - UI_BORDER_SIZE});
}

void ReplayControls::start(const QString &route, const QString &data_dir) {
  QStringList allow = {"modelV2", "controlsState", "liveCalibration", "radarState", "roadCameraState",
                       "roadEncodeIdx", "carParams", "driverMonitoringState", "carState", "liveLocationKalman",
                       "driverStateV2", "wideRoadCameraState", "navInstruction", "navRoute", "uiPlan"};
  QString route_name = "0000000000000000|" + route;
  replay.reset(new Replay(route_name, allow, {}, nullptr, REPLAY_FLAG_QCAMERA | REPLAY_FLAG_LESS_CPU_USAGE, data_dir));
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
