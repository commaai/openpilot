#include <QEventLoop>
#include <QMap>
#include <QThread>

#include "catch2/catch.hpp"
#include "selfdrive/ui/soundd/sound.h"

void controls_thread(int loop_count) {
  PubMaster pm({"carState", "controlsState", "deviceState"});
  MessageBuilder deviceStateMsg;
  auto deviceState = deviceStateMsg.initEvent().initDeviceState();
  deviceState.setStarted(true);

  const int DT_CTRL = 10;  // ms

  // speeds (volume levels)
  const std::vector<float> vEgos = {0, 20, 0, 20,};

  for (float vEgo : vEgos) {
    printf("\n## changing volume to %.1f\n\n", vEgo);
    MessageBuilder carStateMsg;
    auto carState = carStateMsg.initEvent().initCarState();
    carState.setVEgo(vEgo);
    pm.send("carState", carStateMsg);

    for (int i = 0; i < loop_count; ++i) {
      // send no alert sound
      for (int j = 0; j < 1000 / DT_CTRL; ++j) {
        MessageBuilder msg;
        msg.initEvent().initControlsState();
        pm.send("carState", carStateMsg);
        pm.send("controlsState", msg);
        pm.send("deviceState", deviceStateMsg);
        QThread::msleep(DT_CTRL);
      }

      printf("playing engage.wav\n");
      for (int j = 0; j < 1000 / DT_CTRL; ++j) {
        MessageBuilder msg;
        auto cs = msg.initEvent().initControlsState();
        cs.setAlertSound(AudibleAlert::ENGAGE);
        cs.setAlertType("engage.wav");
        pm.send("controlsState", msg);
        pm.send("deviceState", deviceStateMsg);
        QThread::msleep(DT_CTRL);
      }
    }
  }

  QThread::currentThread()->quit();
}

TEST_CASE("test soundd changing volume") {
  QEventLoop loop;
  Sound sound;
  const int test_loop_cnt = 2;

  QThread t;
  QObject::connect(&t, &QThread::started, [=]() { controls_thread(test_loop_cnt); });
  QObject::connect(&t, &QThread::finished, [&]() { loop.quit(); });
  t.start();
  loop.exec();
}
