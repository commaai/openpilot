#include "tools/cabana/streams/replaystream.h"

#include <cstdint>
#include <utility>

#include <QAudioDeviceInfo>
#include <QAudioFormat>
#include <QAudioOutput>
#include <QByteArray>
#include <QDebug>
#include <QIODevice>
#include <QLabel>
#include <QFileDialog>
#include <QGridLayout>
#include <QMessageBox>
#include <QPushButton>

#include "common/timing.h"
#include "common/util.h"
#include "tools/cabana/streams/routes.h"

class ReplayAudioOutput : public QObject {
public:
  void play(const cereal::AudioData::Reader &audio) {
    const auto data = audio.getData();
    const uint32_t sample_rate = audio.getSampleRate();
    if (data.size() == 0 || sample_rate == 0) return;

    QByteArray pcm(reinterpret_cast<const char *>(data.begin()), data.size());
    QMetaObject::invokeMethod(this, [this, pcm = std::move(pcm), sample_rate]() {
      write(sample_rate, pcm);
    }, Qt::QueuedConnection);
  }

  void reset() {
    QMetaObject::invokeMethod(this, [this]() { resetOutput(); }, Qt::QueuedConnection);
  }

private:
  bool ensureOutput(uint32_t sample_rate) {
    if (audio_output && sample_rate == current_sample_rate) return true;
    if (sample_rate == unsupported_sample_rate) return false;

    resetOutput();

    QAudioFormat format;
    format.setSampleRate(sample_rate);
    format.setChannelCount(1);
    format.setSampleSize(16);
    format.setCodec("audio/pcm");
    format.setByteOrder(QAudioFormat::LittleEndian);
    format.setSampleType(QAudioFormat::SignedInt);

    const QAudioDeviceInfo device_info = QAudioDeviceInfo::defaultOutputDevice();
    if (!device_info.isFormatSupported(format)) {
      const QAudioFormat nearest = device_info.nearestFormat(format);
      if (nearest.sampleRate() != format.sampleRate() ||
          nearest.channelCount() != format.channelCount() ||
          nearest.sampleSize() != format.sampleSize() ||
          nearest.sampleType() != format.sampleType()) {
        qWarning() << "rawAudioData playback format is not supported:" << format;
        unsupported_sample_rate = sample_rate;
        return false;
      }
    }

    audio_output = std::make_unique<QAudioOutput>(format);
    audio_output->setBufferSize(sample_rate * sizeof(int16_t) / 2);
    audio_device = audio_output->start();
    current_sample_rate = sample_rate;
    return audio_device != nullptr;
  }

  void write(uint32_t sample_rate, const QByteArray &pcm) {
    if (!ensureOutput(sample_rate)) return;
    if (audio_output->bytesFree() >= pcm.size()) {
      audio_device->write(pcm.constData(), pcm.size());
    }
  }

  void resetOutput() {
    if (audio_output) {
      audio_output->stop();
      audio_output.reset();
    }
    audio_device = nullptr;
    current_sample_rate = 0;
  }

  std::unique_ptr<QAudioOutput> audio_output;
  QIODevice *audio_device = nullptr;
  uint32_t current_sample_rate = 0;
  uint32_t unsupported_sample_rate = 0;
};

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

  audio_output = std::make_unique<ReplayAudioOutput>();
}

ReplayStream::~ReplayStream() {
  replay.reset();
  audio_output.reset();
}

void ReplayStream::mergeSegments() {
  auto event_data = replay->getEventData();
  for (const auto &[n, seg] : event_data->segments) {
    if (!processed_segments.count(n)) {
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
      if (new_events.empty()) {
        static const MessageEventsMap empty_events;
        emit eventsMerged(empty_events);
      } else {
        mergeEvents(new_events);
      }
    }
  }
}

bool ReplayStream::loadRoute(const std::string &route, const std::string &data_dir, uint32_t replay_flags, bool auto_source) {
  audio_output->reset();
  replay.reset(new Replay(route, {"can", "rawAudioData", "roadEncodeIdx", "driverEncodeIdx", "wideRoadEncodeIdx", "carParams"},
                          {}, nullptr, replay_flags, data_dir, auto_source));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter([this](const Event *event) { return eventFilter(event); });

  // Forward replay callbacks to corresponding Qt signals.
  replay->onSeeking = [this](double sec) {
    audio_output->reset();
    emit seeking(sec);
  };
  replay->onSeekedTo = [this](double sec) {
    audio_output->reset();
    emit seekedTo(sec);
    waitForSeekFinshed();
  };
  replay->onQLogLoaded = [this](std::shared_ptr<LogReader> qlog) { emit qLogLoaded(qlog); };
  replay->onSegmentsMerged = [this]() { QMetaObject::invokeMethod(this, &ReplayStream::mergeSegments, Qt::BlockingQueuedConnection); };

  bool success = replay->load();
  if (!success) {
    if (replay->lastRouteError() == RouteLoadError::Unauthorized) {
      auto auth_content = util::read_file(util::getenv("HOME") + "/.comma/auth.json");
      QString message;
      if (auth_content.empty()) {
        message = "Authentication Required. Please run the following command to authenticate:\n\n"
                  "python3 openpilot/tools/lib/auth.py\n\n"
                  "This will grant access to routes from your comma account.";
      } else {
        message = tr("Access Denied. You do not have permission to access route:\n\n%1\n\n"
                     "This is likely a private route.").arg(QString::fromStdString(route));
      }
      QMessageBox::warning(nullptr, tr("Access Denied"), message);
    } else if (replay->lastRouteError() == RouteLoadError::NetworkError) {
      QMessageBox::warning(nullptr, tr("Network Error"),
                          tr("Unable to load the route:\n\n %1.\n\nPlease check your network connection and try again.").arg(QString::fromStdString(route)));
    } else if (replay->lastRouteError() == RouteLoadError::FileNotFound) {
      QMessageBox::warning(nullptr, tr("Route Not Found"),
                           tr("The specified route could not be found:\n\n %1.\n\nPlease check the route name and try again.").arg(QString::fromStdString(route)));
    } else {
      QMessageBox::warning(nullptr, tr("Route Load Failed"), tr("Failed to load route: '%1'").arg(QString::fromStdString(route)));
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
  } else if (event->which == cereal::Event::Which::RAW_AUDIO_DATA && replay->getSpeed() == 1.0f && !replay->isPaused()) {
    capnp::FlatArrayMessageReader reader(event->data);
    auto e = reader.getRoot<cereal::Event>();
    audio_output->play(e.getRawAudioData());
  }

  double ts = millis_since_boot();
  if ((ts - prev_update_ts) > (1000.0 / settings.fps)) {
    emit privateUpdateLastMsgsSignal();
    prev_update_ts = ts;
  }
  return true;
}

void ReplayStream::setSpeed(float speed) {
  replay->setSpeed(speed);
  if (speed != 1.0f) {
    audio_output->reset();
  }
}

void ReplayStream::pause(bool pause) {
  audio_output->reset();
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
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), QString::fromStdString(settings.last_route_dir));
    if (!dir.isEmpty()) {
      route_edit->setText(dir);
      settings.last_route_dir = QFileInfo(dir).absolutePath().toStdString();
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

    if (replay_stream->loadRoute(route.toStdString(), data_dir.toStdString(), flags)) {
      return replay_stream.release();
    }
  }
  return nullptr;
}
