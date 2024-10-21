#include "tools/cabana/streams/replaystream.h"

#include <QLabel>
#include <QFileDialog>
#include <QGridLayout>
#include <QMessageBox>
#include <QPushButton>

#include "common/timing.h"
#include "tools/cabana/streams/routes.h"

ReplayStream::ReplayStream(QObject *parent) : AbstractStream(parent) {
  unsetenv("ZMQ");
  setenv("COMMA_CACHE", "/tmp/comma_download_cache", 1);

  // TODO: Remove when OpenpilotPrefix supports ZMQ
#ifndef __APPLE__
  op_prefix = std::make_unique<OpenpilotPrefix>();
#endif

  QObject::connect(&settings, &Settings::changed, this, [this]() {
    if (replay) replay->setSegmentCacheLimit(settings.max_cached_minutes);
  });
}

static bool event_filter(const Event *e, void *opaque) {
  return ((ReplayStream *)opaque)->eventFilter(e);
}

void ReplayStream::mergeSegments() {
  for (auto &[n, seg] : replay->segments()) {
    if (seg && seg->isLoaded() && !processed_segments.count(n)) {
      processed_segments.insert(n);

      std::vector<const CanEvent *> new_events;
      new_events.reserve(seg->log->events.size());
      for (const Event &e : seg->log->events) {
        if (e.which == cereal::Event::Which::CAN) {
          capnp::FlatArrayMessageReader reader(e.data);
          auto event = reader.getRoot<cereal::Event>();
          for (const auto &c : event.getCan()) {
            new_events.push_back(newEvent(e.mono_time, c));
          }
        }
      }
      mergeEvents(new_events);
    }
  }
}

bool ReplayStream::loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags) {
  replay.reset(new Replay(route.toStdString(), {"can", "roadEncodeIdx", "driverEncodeIdx", "wideRoadEncodeIdx", "carParams"},
                          {}, nullptr, replay_flags, data_dir.toStdString(), this));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay.get(), &Replay::seeking, this, &AbstractStream::seeking);
  QObject::connect(replay.get(), &Replay::seekedTo, this, &AbstractStream::seekedTo);
  QObject::connect(replay.get(), &Replay::segmentsMerged, this, &ReplayStream::mergeSegments);
  bool success = replay->load();
  if (!success) {
    if (replay->lastRouteError() == RouteLoadError::AccessDenied) {
      auto auth_content = util::read_file(util::getenv("HOME") + "/.comma/auth.json");
      QString message;
      if (auth_content.empty()) {
        message = "Authentication Required. Please run the following command to authenticate:\n\n"
                  "python3 tools/lib/auth.py\n\n"
                  "This will grant access to routes from your comma account.";
      } else {
        message = tr("Access Denied. You do not have permission to access route:\n\n%1\n\n"
                     "This is likely a private route.").arg(route);
      }
      QMessageBox::warning(nullptr, tr("Access Denied"), message);
    } else if (replay->lastRouteError() == RouteLoadError::NetworkError) {
      QMessageBox::warning(nullptr, tr("Network Error"),
                          tr("Unable to load the route:\n\n %1.\n\nPlease check your network connection and try again.").arg(route));
    } else if (replay->lastRouteError() == RouteLoadError::FileNotFound) {
      QMessageBox::warning(nullptr, tr("Route Not Found"),
                           tr("The specified route could not be found:\n\n %1.\n\nPlease check the route name and try again.").arg(route));
    } else {
      QMessageBox::warning(nullptr, tr("Route Load Failed"), tr("Failed to load route: '%1'").arg(route));
    }
  }
  return success;
}

bool ReplayStream::eventFilter(const Event *event) {
  static double prev_update_ts = 0;
  if (event->which == cereal::Event::Which::CAN) {
    double current_sec = toSeconds(event->mono_time);
    capnp::FlatArrayMessageReader reader(event->data);
    auto e = reader.getRoot<cereal::Event>();
    for (const auto &c : e.getCan()) {
      MessageId id = {.source = c.getSrc(), .address = c.getAddress()};
      const auto dat = c.getDat();
      updateEvent(id, current_sec, (const uint8_t*)dat.begin(), dat.size());
    }
  }

  double ts = millis_since_boot();
  if ((ts - prev_update_ts) > (1000.0 / settings.fps)) {
    emit privateUpdateLastMsgsSignal();
    prev_update_ts = ts;
  }
  return true;
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  emit(pause ? paused() : resume());
}


// OpenReplayWidget

OpenReplayWidget::OpenReplayWidget(QWidget *parent) : AbstractOpenStreamWidget(parent) {
  QGridLayout *grid_layout = new QGridLayout(this);
  grid_layout->addWidget(new QLabel(tr("Route")), 0, 0);
  grid_layout->addWidget(route_edit = new QLineEdit(this), 0, 1);
  route_edit->setPlaceholderText(tr("Enter route name or browse for local/remote route"));
  auto browse_remote_btn = new QPushButton(tr("Remote route..."), this);
  grid_layout->addWidget(browse_remote_btn, 0, 2);
  auto browse_local_btn = new QPushButton(tr("Local route..."), this);
  grid_layout->addWidget(browse_local_btn, 0, 3);

  QHBoxLayout *camera_layout = new QHBoxLayout();
  for (auto c : {tr("Road camera"), tr("Driver camera"), tr("Wide road camera")})
    camera_layout->addWidget(cameras.emplace_back(new QCheckBox(c, this)));
  cameras[0]->setChecked(true);
  camera_layout->addStretch(1);
  grid_layout->addItem(camera_layout, 1, 1);

  setMinimumWidth(550);
  QObject::connect(browse_local_btn, &QPushButton::clicked, [=]() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), settings.last_route_dir);
    if (!dir.isEmpty()) {
      route_edit->setText(dir);
      settings.last_route_dir = QFileInfo(dir).absolutePath();
    }
  });
  QObject::connect(browse_remote_btn, &QPushButton::clicked, [this]() {
    RoutesDialog route_dlg(this);
    if (route_dlg.exec()) {
      route_edit->setText(route_dlg.route());
    }
  });
}

AbstractStream *OpenReplayWidget::open() {
  QString route = route_edit->text();
  QString data_dir;
  if (int idx = route.lastIndexOf('/'); idx != -1 && util::file_exists(route.toStdString())) {
    data_dir = route.mid(0, idx + 1);
    route = route.mid(idx + 1);
  }

  bool is_valid_format = Route::parseRoute(route.toStdString()).str.size() > 0;
  if (!is_valid_format) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Invalid route format: '%1'").arg(route));
  } else {
    auto replay_stream = std::make_unique<ReplayStream>(qApp);
    uint32_t flags = REPLAY_FLAG_NONE;
    if (cameras[1]->isChecked()) flags |= REPLAY_FLAG_DCAM;
    if (cameras[2]->isChecked()) flags |= REPLAY_FLAG_ECAM;
    if (flags == REPLAY_FLAG_NONE && !cameras[0]->isChecked()) flags = REPLAY_FLAG_NO_VIPC;

    if (replay_stream->loadRoute(route, data_dir, flags)) {
      return replay_stream.release();
    }
  }
  return nullptr;
}
