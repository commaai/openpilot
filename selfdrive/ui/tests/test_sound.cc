#include <QEventLoop>
#include <QMap>
#include <QThread>

#include "catch2/catch.hpp"
#include "selfdrive/ui/soundd/sound.h"

const int test_loop_cnt = 2;

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
    QEventLoop loop;
    QThread t;
    QObject::connect(&t, &QThread::started, [=]() { controls_thread(test_loop_cnt); });
    QObject::connect(&t, &QThread::finished, [&]() { loop.quit(); });
    t.start();
    loop.exec();
  }

  void controls_thread(int loop_cnt);
  QMap<AudibleAlert, std::pair<int, int>> sound_stats;
};

void TestSound::controls_thread(int loop_cnt) {
  PubMaster pm({"controlsState", "deviceState"});
  MessageBuilder deviceStateMsg;
  auto deviceState = deviceStateMsg.initEvent().initDeviceState();
  deviceState.setStarted(true);

  const int DT_CTRL = 10;  // ms
  for (int i = 0; i < loop_cnt; ++i) {
    for (auto it = sounds.begin(); it != sounds.end(); ++it) {
      auto &s = it.value();
      printf("testing %s\n", qPrintable(s.file));
      for (int j = 0; j < (s.loops_to_full_volume > 0 ? s.loops_to_full_volume * 1000 : 1000) / DT_CTRL; ++j) {
        MessageBuilder msg;
        auto cs = msg.initEvent().initControlsState();
        cs.setAlertSound(it.key());
        cs.setAlertType(s.file);
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

  for (const AudibleAlert alert : sound_stats.keys()) {
    auto [play, stop] = sound_stats[alert];
    REQUIRE(play == test_loop_cnt);
    REQUIRE(stop == test_loop_cnt);
  }

  QThread::currentThread()->quit();
}

TEST_CASE("test soundd") {
  TestSound test_sound;
}
