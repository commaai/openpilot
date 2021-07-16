#include <map>

#include <QApplication>
#include <QString>
#include <QSoundEffect>

#include "cereal/gen/cpp/log.capnp.h"
#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

// TODO: handle controls unresponsive alert
// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

class Sound : public QObject {

public:
  explicit Sound(QObject *parent = 0) {
    sm = new SubMaster({"carState", "controlsState"});

    std::tuple<AudibleAlert, QString, bool> sound_list[] = {
      {AudibleAlert::CHIME_DISENGAGE, "../assets/sounds/disengaged.wav", false},
      {AudibleAlert::CHIME_ENGAGE, "../assets/sounds/engaged.wav", false},
      {AudibleAlert::CHIME_WARNING1, "../assets/sounds/warning_1.wav", false},
      {AudibleAlert::CHIME_WARNING2, "../assets/sounds/warning_2.wav", false},
      {AudibleAlert::CHIME_WARNING2_REPEAT, "../assets/sounds/warning_2.wav", true},
      {AudibleAlert::CHIME_WARNING_REPEAT, "../assets/sounds/warning_repeat.wav", true},
      {AudibleAlert::CHIME_ERROR, "../assets/sounds/error.wav", false},
      {AudibleAlert::CHIME_PROMPT, "../assets/sounds/error.wav", false}
    };

    for (auto &[alert, fn, loops] : sound_list) {
      sounds[alert].first.setSource(QUrl::fromLocalFile(fn));
      sounds[alert].second = loops ? QSoundEffect::Infinite : 0;
    }

    QTimer *timer = new QTimer(this);
    QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
    timer->start();
  };
  ~Sound() {
    delete sm;
  };

  Alert alert;
  float volume = Hardware::MIN_VOLUME;
  std::map<AudibleAlert, std::pair<QSoundEffect, int>> sounds;
  SubMaster *sm;

private slots:
  void update() {
    sm->update(100);
    if (sm->updated("carState")) {
      // scale volume with speed
      volume = util::map_val((*sm)["carState"].getCarState().getVEgo(), 0.f, 20.f,
                             Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
    }
    if (sm->updated("controlsState")) {
      // TODO: properly check if alert changed
      const cereal::ControlsState::Reader &cs = (*sm)["controlsState"].getControlsState();
      Alert latest = {QString::fromStdString(cs.getAlertText1()),
                      QString::fromStdString(cs.getAlertText2()),
                      QString::fromStdString(cs.getAlertType()),
                      cs.getAlertSize(), cs.getAlertSound()};
      if (!alert.equal(latest)) {
        alert = latest;
        // stop sounds
        for (auto &kv : sounds) {
          // Only stop repeating sounds
          auto &[sound, loops] = kv.second;
          if (sound.loopsRemaining() == QSoundEffect::Infinite) {
            sound.stop();
          }
        }

        if (alert.sound != AudibleAlert::NONE) {
          auto &[sound, loops] = sounds[alert.sound];
          sound.setLoopCount(loops);
          sound.setVolume(volume);
          sound.play();
        }
      }
    }
  }
};

int main(int argc, char **argv) {
  // TODO: priority? can't mismatch UI too much
  QApplication a(argc, argv);
  Sound sound;
  return a.exec();
}
