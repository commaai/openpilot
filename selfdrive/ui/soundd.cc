#include <sys/resource.h>

#include <map>

#include <QApplication>
#include <QString>
#include <QSoundEffect>

#include "selfdrive/ui/qt/util.h"
#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

void sigHandler(int s) {
  qApp->quit();
}

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0) : sm({"carState", "controlsState"}) {
    // TODO: merge again and add EQ in the amp config
    const QString sound_asset_path = Hardware::TICI() ? "../assets/sounds_tici/" : "../assets/sounds/";
    std::tuple<AudibleAlert, QString, bool> sound_list[] = {
        {AudibleAlert::CHIME_DISENGAGE, "disengaged.wav", false},
        {AudibleAlert::CHIME_ENGAGE, "engaged.wav", false},
        {AudibleAlert::CHIME_WARNING1, "warning_1.wav", false},
        {AudibleAlert::CHIME_WARNING2, "warning_2.wav", false},
        {AudibleAlert::CHIME_WARNING2_REPEAT, "warning_2.wav", true},
        {AudibleAlert::CHIME_WARNING_REPEAT, "warning_repeat.wav", true},
        {AudibleAlert::CHIME_ERROR, "error.wav", false},
        {AudibleAlert::CHIME_PROMPT, "error.wav", false}};
    for (auto &[alert, fn, loops] : sound_list) {
      QSoundEffect *s = new QSoundEffect(this);
      QObject::connect(s, &QSoundEffect::statusChanged, [=]() { assert(s->status() != QSoundEffect::Error); });
      s->setSource(QUrl::fromLocalFile(sound_asset_path + fn));
      sounds[alert] = {s, loops ? QSoundEffect::Infinite : 0};
    }

    QTimer *timer = new QTimer(this);
    QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
    timer->start();
  };

  void update() {
    sm.update(100);
    if (sm.updated("carState")) {
      // scale volume with speed
      float volume = util::map_val(sm["carState"].getCarState().getVEgo(), 0.f, 20.f,
                                   Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
      for (auto &[s, loops] : sounds) {
        s->setVolume(std::round(100 * volume) / 100);
      }
    }
    if (sm.updated("controlsState")) {
      setAlert(sm["controlsState"].getControlsState().getAlertSound());
    } else if (sm.rcv_frame("controlsState") > 0 && sm["controlsState"].getControlsState().getEnabled() &&
               ((nanos_since_boot() - sm.rcv_time("controlsState")) / 1e9 > CONTROLS_TIMEOUT)) {
      setAlert(CONTROLS_UNRESPONSIVE_ALERT.sound);
    }
  }

  void setAlert(AudibleAlert sound) {
    if (current_sound_ != sound) {
      current_sound_ = sound;
      // stop sounds
      for (auto &[s, loops] : sounds) {
        // Only stop repeating sounds
        if (s->loopsRemaining() == QSoundEffect::Infinite) {
          s->stop();
        }
      }

      // play sound
      if (sound != AudibleAlert::NONE) {
        auto &[s, loops] = sounds[sound];
        s->setLoopCount(loops);
        s->play();
      }
    }
  }

private:
  AudibleAlert current_sound_ = AudibleAlert::NONE;
  QMap<AudibleAlert, QPair<QSoundEffect*, int>> sounds;
  SubMaster sm;
};

int main(int argc, char **argv) {
  qInstallMessageHandler(swagLogMessageHandler);
  setpriority(PRIO_PROCESS, 0, -20);

  QApplication a(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

  Sound sound;
  return a.exec();
}
