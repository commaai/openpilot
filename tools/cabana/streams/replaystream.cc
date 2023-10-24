#include "tools/cabana/streams/replaystream.h"

#include <QLabel>
#include <QFileDialog>
#include <QGridLayout>
#include <QMessageBox>
#include <QPushButton>

ReplayStream::ReplayStream(QObject *parent) : AbstractStream(parent) {
  unsetenv("ZMQ");
  setenv("COMMA_CACHE", "/tmp/comma_download_cache", 1);

  // TODO: Remove when OpenpilotPrefix supports ZMQ
#ifndef __APPLE__
  op_prefix = std::make_unique<OpenpilotPrefix>();
#endif

  QObject::connect(&settings, &Settings::changed, [this]() {
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
      const auto &events = seg->log->events;
      mergeEvents(events.cbegin(), events.cend());
    }
  }
}

bool ReplayStream::loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags) {
  replay.reset(new Replay(route, {"can", "roadEncodeIdx", "wideRoadEncodeIdx", "carParams"}, {}, {}, nullptr, replay_flags, data_dir, this));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay.get(), &Replay::seekedTo, this, &AbstractStream::seekedTo);
  QObject::connect(replay.get(), &Replay::segmentsMerged, this, &ReplayStream::mergeSegments);
  QObject::connect(replay.get(), &Replay::qLogLoaded, this, &ReplayStream::qLogLoaded, Qt::QueuedConnection);
  return replay->load();
}

void ReplayStream::start() {
  emit streamStarted();
  replay->start();
}

bool ReplayStream::eventFilter(const Event *event) {
  static double prev_update_ts = 0;
  // delay posting CAN message if UI thread is busy
  if (event->which == cereal::Event::Which::CAN) {
    double current_sec = event->mono_time / 1e9 - routeStartTime();
    for (const auto &c : event->event.getCan()) {
      MessageId id = {.source = c.getSrc(), .address = c.getAddress()};
      const auto dat = c.getDat();
      updateEvent(id, current_sec, (const uint8_t*)dat.begin(), dat.size());
    }
  }

  double ts = millis_since_boot();
  if ((ts - prev_update_ts) > (1000.0 / settings.fps)) {
    if (postEvents()) {
      prev_update_ts = ts;
    }
  }
  return true;
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  emit(pause ? paused() : resume());
}


AbstractOpenStreamWidget *ReplayStream::widget(AbstractStream **stream) {
  return new OpenReplayWidget(stream);
}

// OpenReplayWidget

OpenReplayWidget::OpenReplayWidget(AbstractStream **stream) : AbstractOpenStreamWidget(stream) {
  // TODO: get route list from api.comma.ai
  QGridLayout *grid_layout = new QGridLayout();
  grid_layout->addWidget(new QLabel(tr("Route")), 0, 0);
  grid_layout->addWidget(route_edit = new QLineEdit(this), 0, 1);
  route_edit->setPlaceholderText(tr("Enter remote route name or click browse to select a local route"));
  auto file_btn = new QPushButton(tr("Browse..."), this);
  grid_layout->addWidget(file_btn, 0, 2);

  grid_layout->addWidget(new QLabel(tr("Video")), 1, 0);
  grid_layout->addWidget(choose_video_cb = new QComboBox(this), 1, 1);
  QString items[] = {tr("No Video"), tr("Road Camera"), tr("Wide Road Camera"), tr("Driver Camera"), tr("QCamera")};
  for (int i = 0; i < std::size(items); ++i) {
    choose_video_cb->addItem(items[i]);
  }
  choose_video_cb->setCurrentIndex(1);  // default is road camera;

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(grid_layout);
  setMinimumWidth(550);

  QObject::connect(file_btn, &QPushButton::clicked, [=]() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), settings.last_route_dir);
    if (!dir.isEmpty()) {
      route_edit->setText(dir);
      settings.last_route_dir = QFileInfo(dir).absolutePath();
    }
  });
}

bool OpenReplayWidget::open() {
  QString route = route_edit->text();
  QString data_dir;
  if (int idx = route.lastIndexOf('/'); idx != -1) {
    data_dir = route.mid(0, idx + 1);
    route = route.mid(idx + 1);
  }

  bool is_valid_format = Route::parseRoute(route).str.size() > 0;
  if (!is_valid_format) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Invalid route format: '%1'").arg(route));
  } else {
    uint32_t flags[] = {REPLAY_FLAG_NO_VIPC, REPLAY_FLAG_NONE, REPLAY_FLAG_ECAM, REPLAY_FLAG_DCAM, REPLAY_FLAG_QCAMERA};
    auto replay_stream = std::make_unique<ReplayStream>(qApp);
    if (replay_stream->loadRoute(route, data_dir, flags[choose_video_cb->currentIndex()])) {
      *stream = replay_stream.release();
    } else {
      QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to load route: '%1'").arg(route));
    }
  }
  return *stream != nullptr;
}
