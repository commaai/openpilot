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
  explicit Sound(QObject *parent = 0) {
    // TODO: merge again and add EQ in the amp config
    const QString sound_asset_path = Hardware::TICI() ? "../assets/sounds_tici/" : "../assets/sounds/";
    std::tuple<AudibleAlert, QString, bool> sound_list[] = {
      {AudibleAlert::CHIME_DISENGAGE, sound_asset_path + "disengaged.wav", false},
      {AudibleAlert::CHIME_ENGAGE, sound_asset_path + "engaged.wav", false},
      {AudibleAlert::CHIME_WARNING1, sound_asset_path + "warning_1.wav", false},
      {AudibleAlert::CHIME_WARNING2, sound_asset_path + "warning_2.wav", false},
      {AudibleAlert::CHIME_WARNING2_REPEAT, sound_asset_path + "warning_2.wav", true},
      {AudibleAlert::CHIME_WARNING_REPEAT, sound_asset_path + "warning_repeat.wav", true},
      {AudibleAlert::CHIME_ERROR, sound_asset_path + "error.wav", false},
      {AudibleAlert::CHIME_PROMPT, sound_asset_path + "error.wav", false}
    };
    for (auto &[alert, fn, loops] : sound_list) {
      QSoundEffect *s = new QSoundEffect(this);
      QObject::connect(s, &QSoundEffect::statusChanged, this, &Sound::checkStatus);
      s->setSource(QUrl::fromLocalFile(fn));
      sounds[alert] = {s, loops ? QSoundEffect::Infinite : 0};
    }

    sm = new SubMaster({"carState", "controlsState"});

    QTimer *timer = new QTimer(this);
    QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
    timer->start();
  };
  ~Sound() {
    delete sm;
  };

private slots:
  void checkStatus() {
    QSoundEffect *s = qobject_cast<QSoundEffect*>(sender());
    assert(s->status() != QSoundEffect::Error);
  }

  void update() {
    sm->update(100);
    if (sm->updated("carState")) {
      // scale volume with speed
      volume = util::map_val((*sm)["carState"].getCarState().getVEgo(), 0.f, 20.f,
                             Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
    }
    if (sm->updated("controlsState")) {
      const cereal::ControlsState::Reader &cs = (*sm)["controlsState"].getControlsState();
      setAlert({QString::fromStdString(cs.getAlertText1()),
                QString::fromStdString(cs.getAlertText2()),
                QString::fromStdString(cs.getAlertType()),
                cs.getAlertSize(), cs.getAlertSound()});
    } else if (sm->rcv_frame("controlsState") > 0 && (*sm)["controlsState"].getControlsState().getEnabled() &&
               ((nanos_since_boot() - sm->rcv_time("controlsState")) / 1e9 > CONTROLS_TIMEOUT)) {
      setAlert(CONTROLS_UNRESPONSIVE_ALERT);
    }
  }

  void setAlert(Alert a) {
    if (!alert.equal(a)) {
      alert = a;
      // stop sounds
      for (auto &[s, loops] : sounds) {
        // Only stop repeating sounds
        if (s->loopsRemaining() == QSoundEffect::Infinite) {
          s->stop();
        }
      }

      // play sound
      if (alert.sound != AudibleAlert::NONE) {
        auto &[s, loops] = sounds[alert.sound];
        s->setLoopCount(loops);
        s->setVolume(volume);
        s->play();
      }
    }
  }

private:
  Alert alert;
  float volume = Hardware::MIN_VOLUME;
  QMap<AudibleAlert, QPair<QSoundEffect*, int>> sounds;
  SubMaster *sm;
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
