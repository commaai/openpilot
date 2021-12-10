#include <QEventLoop>
#include <QMap>
#include <QThread>

#include "catch2/catch.hpp"
#include "selfdrive/ui/soundd/sound.h"

class TestSound : public Sound {
public:
  TestSound() : Sound() {
    for (auto i = sounds.constBegin(); i != sounds.constEnd(); ++i) {
      sound_stats[i.key()] = {0, 0};
      auto &s = i.value();
      QObject::connect(s.sound, &QSoundEffect::playingChanged, [this, &s, a = i.key()]() {
        if (s.sound->isPlaying()) {
          sound_stats[a].first++;
          REQUIRE(s.sound->volume() == current_volume);
        } else {
          sound_stats[a].second++;
          if (s.loops_to_full_volume > 0) {
            REQUIRE(s.sound->volume() == 1.0);
          }
        }
      });
    }
  }

  QMap<AudibleAlert, std::pair<int, int>> sound_stats;
};

void controls_thread(int loop_cnt) {
  PubMaster pm({"controlsState", "deviceState"});
  MessageBuilder deviceStateMsg;
  auto deviceState = deviceStateMsg.initEvent().initDeviceState();
  deviceState.setStarted(true);

  const int DT_CTRL = 10;  // ms
  for (int i = 0; i < loop_cnt; ++i) {
    for (auto &[alert, fn, loops, loops_to_max_volume] : sound_list) {
      printf("testing %s\n", qPrintable(fn));
      for (int j = 0; j < (loops_to_max_volume > 0 ? loops_to_max_volume * 1000 : 1000) / DT_CTRL; ++j) {
        MessageBuilder msg;
        auto cs = msg.initEvent().initControlsState();
        cs.setAlertSound(alert);
        cs.setAlertType(fn);
        pm.send("controlsState", msg);
        pm.send("deviceState", deviceStateMsg);
        QThread::msleep(DT_CTRL);
      }
    }
  }

  // send no alert sound
  for (int j = 0; j < 1000 / DT_CTRL; ++j) {
    MessageBuilder msg;
    msg.initEvent().initControlsState();
    pm.send("controlsState", msg);
    QThread::msleep(DT_CTRL);
  }

  QThread::currentThread()->quit();
}

TEST_CASE("test soundd") {
  QEventLoop loop;
  TestSound test_sound;
  const int test_loop_cnt = 2;

  QThread t;
  QObject::connect(&t, &QThread::started, [=]() { controls_thread(test_loop_cnt); });
  QObject::connect(&t, &QThread::finished, [&]() { loop.quit(); });
  t.start();
  loop.exec();

  for (const AudibleAlert alert : test_sound.sound_stats.keys()) {
    auto [play, stop] = test_sound.sound_stats[alert];
    REQUIRE(play == test_loop_cnt);
    REQUIRE(stop == test_loop_cnt);
  }
}
