#include "selfdrive/ui/soundd/sound.h"

#include <cmath>

#include <QAudio>
#include <QAudioDeviceInfo>
#include <QAudioProbe>
#include <QAudioRecorder>
#include <QDebug>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"

// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

Sound::Sound(QObject *parent) : sm({"carState", "controlsState", "deviceState"}) {
  qInfo() << "default audio device: " << QAudioDeviceInfo::defaultOutputDevice().deviceName();

  // setup alert sounds
  for (auto &[alert, fn, loops] : sound_list) {
    QSoundEffect *s = new QSoundEffect(this);
    QObject::connect(s, &QSoundEffect::statusChanged, [=]() {
      assert(s->status() != QSoundEffect::Error);
    });
    s->setVolume(Hardware::MIN_VOLUME);
    s->setSource(QUrl::fromLocalFile("../../assets/sounds/" + fn));
    sounds[alert] = {s, loops};
  }

  // setup recording
  QAudioProbe *probe = new QAudioProbe(this);
  QAudioRecorder *recorder = new QAudioRecorder(this);
  {
    qDebug() << recorder->supportedAudioCodecs();

    bool ret = probe->setSource(recorder);
    assert(ret);
    connect(probe, &QAudioProbe::audioBufferProbed, this, &Sound::handleRecording);

    QAudioEncoderSettings audioSettings;
    audioSettings.setCodec("audio/AMR");
    audioSettings.setQuality(QMultimedia::HighQuality);

    recorder->setAudioSettings(audioSettings);
    recorder->record();
  }

  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
  timer->start(1000 / UI_FREQ);
};

void Sound::handleRecording(const QAudioBuffer buf) {
  // ambient sound

  qDebug() << "got recording" << buf.format() << buf.isValid() << buf.sampleCount() << buf.startTime() << buf.duration();
}

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

  setAlert(Alert::get(sm, started_frame));
}

void Sound::setAlert(const Alert &alert) {
  if (!current_alert.equal(alert)) {
    current_alert = alert;
    // stop sounds
    for (auto &[s, loops] : sounds) {
      // Only stop repeating sounds
      if (s->loopsRemaining() > 1 || s->loopsRemaining() == QSoundEffect::Infinite) {
        s->setLoopCount(0);
      }
    }

    // play sound
    if (alert.sound != AudibleAlert::NONE) {
      auto &[s, loops] = sounds[alert.sound];
      s->setLoopCount(loops);
      s->play();
    }
  }
}
