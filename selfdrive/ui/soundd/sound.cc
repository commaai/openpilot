#include "selfdrive/ui/soundd/sound.h"

#include <cmath>

#include <QAudio>
#include <QAudioDeviceInfo>
#include <QDebug>

#include "cereal/messaging/messaging.h"
#include "common/util.h"

// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

Sound::Sound(QObject *parent) : sm({"controlsState", "deviceState", "microphone"}) {
  qInfo() << "default audio device: " << QAudioDeviceInfo::defaultOutputDevice().deviceName();

  for (auto &[alert, fn, loops] : sound_list) {
    QSoundEffect *s = new QSoundEffect(this);
    QObject::connect(s, &QSoundEffect::statusChanged, [=]() {
      assert(s->status() != QSoundEffect::Error);
    });
    s->setSource(QUrl::fromLocalFile("../../assets/sounds/" + fn));
    sounds[alert] = {s, loops};
  }

  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
  timer->start(1000 / UI_FREQ);
};

void Sound::update() {
  const bool started_prev = sm["deviceState"].getDeviceState().getStarted();
  sm.update(0);

  const bool started = sm["deviceState"].getDeviceState().getStarted();
  if (started && !started_prev) {
    started_frame = sm.frame;
  }

  // no sounds while offroad
  // also no sounds if nothing is alive in case thermald crashes while offroad
  const bool crashed = (sm.frame - std::max(sm.rcv_frame("deviceState"), sm.rcv_frame("controlsState"))) > 10*UI_FREQ;
  if (!started || crashed) {
    setAlert({});
    return;
  }

  // scale volume with measured sound level
  if (sm.updated("microphone")) {
    float volume = util::map_val(sm["microphone"].getMicrophone().getFilteredSoundPressureDb(), 58.f, 77.f, 0.f, 1.f);
    current_volume = QAudio::convertVolume(volume, QAudio::LogarithmicVolumeScale, QAudio::LinearVolumeScale);
  }

  setAlert(Alert::get(sm, started_frame));
}

void Sound::setAlert(const Alert &alert) {
  if (!current_alert.equal(alert)) {
    current_alert = alert;
    // stop sounds
    for (auto &[s, loops] : sounds) {
      // Only stop repeating sounds
      if (s->loopsRemaining() > 1 || s->loopsRemaining() == QSoundEffect::Infinite) {
        s->stop();
      }
    }

    // play sound
    if (alert.sound != AudibleAlert::NONE) {
      current_start_time = millis_since_boot();

      auto &[s, loops] = sounds[alert.sound];
      updateVolume(loops != 0);
      s->setLoopCount(loops);
      s->play();
    }
  } else if (alert.sound != AudibleAlert::NONE) {
    auto &[s, loops] = sounds[alert.sound];
    updateVolume(loops != 0);
  }
}

void Sound::updateVolume(bool ramp_up) {
  if (ramp_up) {
    float elapsed_time = (millis_since_boot() - current_start_time) / 1000.0f;
    float volume = util::map_val(elapsed_time, 0.0f, 3.0f, current_volume - 0.4f, current_volume);
    Hardware::set_volume(volume);
  } else {
    Hardware::set_volume(current_volume);
  }
}
