#include <QEventLoop>
#include <QMap>
#include <QThread>

#include "catch2/catch.hpp"
#include "selfdrive/ui/soundd/sound.h"

class TestSound : public Sound {
public:
  TestSound() : Sound() {
    for (auto i = sounds.constBegin(); i != sounds.constEnd(); ++i) {
      QObject::connect(i.value().first, &QSoundEffect::playingChanged, [=, s = i.value().first, a = i.key()]() {
        if (s->isPlaying()) {
          bool repeat = a == AudibleAlert::CHIME_WARNING_REPEAT || a == AudibleAlert::CHIME_WARNING2_REPEAT;
          REQUIRE((s->loopsRemaining() == repeat ? QSoundEffect::Infinite : 1));
          sound_stats[a].first++;
        } else {
          sound_stats[a].second++;
        }
      });
    }
  }

  QMap<AudibleAlert, std::pair<int, int>> sound_stats;
};

void controls_thread(int loop_cnt) {
  PubMaster pm({"controlsState"});
  const int DT_CTRL = 10;  // ms
  for (int i = 0; i < loop_cnt; ++i) {
    for (auto &[alert, fn, loops] : sound_list) {
      printf("testing %s\n", qPrintable(fn));
      for (int j = 0; j < 1000 / DT_CTRL; ++j) {
        MessageBuilder msg;
        auto cs = msg.initEvent().initControlsState();
        cs.setAlertSound(alert);
        cs.setAlertType(fn.toStdString());
        pm.send("controlsState", msg);
        QThread::msleep(DT_CTRL);
      }
    }
  }
  QThread::currentThread()->quit();
}

TEST_CASE("test sound") {
  QEventLoop loop;

  TestSound test_sound;

  const int test_loop_cnt = 2;
  QThread t;
  QObject::connect(&t, &QThread::started, [=]() { controls_thread(test_loop_cnt); });
  QObject::connect(&t, &QThread::finished, [&]() { loop.quit(); });
  t.start();

  loop.exec();

  for (auto [play, stop] : test_sound.sound_stats) {
    REQUIRE(play == test_loop_cnt);
    REQUIRE(stop == test_loop_cnt);
  }
}
