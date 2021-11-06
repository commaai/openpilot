#include "selfdrive/ui/soundd/sound.h"

// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

Sound::Sound(QObject *parent) : sm({"carState", "controlsState"}) {
  // TODO: merge again and add EQ in the amp config
  const QString sound_asset_path = Hardware::TICI() ? "../../assets/sounds_tici/" : "../../assets/sounds/";
  for (auto &[alert, fn, loops] : sound_list) {
    QSoundEffect *s = new QSoundEffect(this);
    QObject::connect(s, &QSoundEffect::statusChanged, [=]() {
      assert(s->status() != QSoundEffect::Error);
    });
    s->setSource(QUrl::fromLocalFile(sound_asset_path + fn));
    s->setVolume(current_volume);
    sounds[alert] = {s, loops ? QSoundEffect::Infinite : 0};
  }

  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
  timer->start(1000 / UI_FREQ);
};

void Sound::update() {
  sm.update(0);
  if (sm.updated("carState")) {
    // scale volume with speed
    float volume = util::map_val(sm["carState"].getCarState().getVEgo(), 0.f, 20.f,
                                 Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
    if (current_volume != volume) {
      current_volume = volume;
      for (auto &[s, loops] : sounds) {
        s->setVolume(std::round(100 * volume) / 100);
      }
    }
  }

  if (auto alert = Alert::get(sm, 1)) {
    setAlert(alert->type, alert->sound);
  } else {
    setAlert({}, AudibleAlert::NONE);
  }
}

void Sound::setAlert(const QString &alert_type, AudibleAlert sound) {
  if (alert_type != current_alert_type || current_sound != sound) {
    current_alert_type = alert_type;
    current_sound = sound;
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

// const int test_loop_cnt = 2;

// void test_sound() {
//   PubMaster pm({"controlsState"});
//   const int DT_CTRL = 10;  // ms
//   for (int i = 0; i < test_loop_cnt; ++i) {
//     for (auto &[alert, fn, loops] : sound_list) {
//       printf("testing %s\n", qPrintable(fn));
//       for (int j = 0; j < 1000 / DT_CTRL; ++j) {
//         MessageBuilder msg;
//         auto cs = msg.initEvent().initControlsState();
//         cs.setAlertSound(alert);
//         cs.setAlertType(fn.toStdString());
//         pm.send("controlsState", msg);
//         QThread::msleep(DT_CTRL);
//       }
//     }
//   }
//   QThread::currentThread()->quit();
// }

// void run_test(Sound *sound) {
//   static QMap<AudibleAlert, std::pair<int, int>> stats;
//   for (auto i = sound->sounds.constBegin(); i != sound->sounds.constEnd(); ++i) {
//     QObject::connect(i.value().first, &QSoundEffect::playingChanged, [s = i.value().first, a = i.key()]() {
//       if (s->isPlaying()) {
//         bool repeat = a == AudibleAlert::CHIME_WARNING_REPEAT || a == AudibleAlert::CHIME_WARNING2_REPEAT;
//         assert(s->loopsRemaining() == repeat ? QSoundEffect::Infinite : 1);
//         stats[a].first++;
//       } else {
//         stats[a].second++;
//       }
//     });
//   }

//   QThread *t = new QThread(qApp);
//   QObject::connect(t, &QThread::started, [=]() { test_sound(); });
//   QObject::connect(t, &QThread::finished, [&]() {
//     for (auto [play, stop] : stats) {
//       assert(play == test_loop_cnt && stop == test_loop_cnt);
//     }
//     qApp->quit();
//   });
//   t->start();
// }
