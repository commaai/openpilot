#include "selfdrive/ui/soundd/sound.h"

#include <cmath>

#include <QAudio>
#include <QAudioDeviceInfo>
#include <QDebug>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

Sound::Sound(QObject *parent) : current_volume(Hardware::MIN_VOLUME), sm({"carState", "controlsState", "deviceState"}) {
  qInfo() << "default audio device: " << QAudioDeviceInfo::defaultOutputDevice().deviceName();

  for (auto &s : sounds) {
    s.sound = new QSoundEffect(this);
    s.sound->setVolume(current_volume);
    s.sound->setSource(QUrl::fromLocalFile(QString("../../assets/sounds/") + s.file));

    QObject::connect(s.sound, &QSoundEffect::statusChanged, [=]() {
      assert(s.sound->status() != QSoundEffect::Error);
    });
    if (s.loops_to_full_volume > 0) {
      QObject::connect(s.sound, &QSoundEffect::loopsRemainingChanged, [this, &s]() { updateVolume(s); });
    }
  }

  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
  timer->start(1000 / UI_FREQ);
}

void Sound::update() {
  sm.update(0);

  const bool started = sm["deviceState"].getDeviceState().getStarted();
  if (started && !started_prev) {
    started_frame = sm.frame;
  }
  started_prev = started;

  // no sounds while offroad
  // also no sounds if nothing is alive in case thermald crashes while offroad
  const bool crashed = (sm.frame - std::max(sm.rcv_frame("deviceState"), sm.rcv_frame("controlsState"))) > 10*UI_FREQ;
  if (!started || crashed) {
    setAlert({});
    return;
  }

  // scale volume with speed
  if (sm.updated("carState")) {
    float volume = util::map_val(sm["carState"].getCarState().getVEgo(), 11.f, 20.f, 0.f, 1.0f);
    volume = QAudio::convertVolume(volume, QAudio::LogarithmicVolumeScale, QAudio::LinearVolumeScale);
    volume = util::map_val(volume, 0.f, 1.f, Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
    current_volume = std::round(100 * volume) / 100;
  }

  setAlert(Alert::get(sm, started_frame));
}

void Sound::setAlert(const Alert &alert) {
  if (!current_alert.equal(alert)) {
    current_alert = alert;
    // stop sounds
    for (const auto &s : sounds) {
      // Only stop repeating sounds
      if (s.sound->loopsRemaining() > 0) {
        s.sound->stop();
      }
    }

    // play sound
    if (alert.sound != AudibleAlert::NONE) {
      const auto &s = sounds[alert.sound];
      s.sound->setVolume(current_volume);
      s.sound->setLoopCount(s.loops);
      s.sound->play();
    }
  }
}

void Sound::updateVolume(const SoundItem &s) {
  int looped = s.sound->loopCount() - s.sound->loopsRemaining();
  if (looped == 0) return;

  qreal volume = s.sound->volume();
  if (looped >= s.loops_to_full_volume) {
    volume = 1.0;
  } else {
    volume += (1.0 - volume) / (s.loops_to_full_volume - looped);
  }
  s.sound->setVolume(std::min(1.0, volume));
}
