#include <QApplication>
#include <QtWidgets>
#include <QTimer>

#include "cereal/messaging/messaging.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/qt_window.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget w;
  setMainWindow(&w);

  w.setStyleSheet("background-color: black;");

  // our beautiful UI
  QVBoxLayout *layout = new QVBoxLayout(&w);
  QLabel *label = new QLabel("〇");
  layout->addWidget(label, 0, Qt::AlignCenter);

  QTimer timer;
  QObject::connect(&timer, &QTimer::timeout, [=]() {
    static SubMaster sm({"deviceState", "controlsState"});

    bool onroad_prev = sm.allAliveAndValid({"deviceState"}) &&
                       sm["deviceState"].getDeviceState().getStarted();
    sm.update(0);

    bool onroad = sm.allAliveAndValid({"deviceState"}) &&
                  sm["deviceState"].getDeviceState().getStarted();

    if (onroad) {
      auto cs = sm["controlsState"].getControlsState();
      UIStatus status = cs.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED;
      if (cs.getAlertStatus() == cereal::ControlsState::AlertStatus::USER_PROMPT) {
        status = STATUS_WARNING;
      } else if (cs.getAlertStatus() == cereal::ControlsState::AlertStatus::CRITICAL) {
        status = STATUS_ALERT;
      }
      label->setStyleSheet(QString("color: %1; font-size: 250px;").arg(bg_colors[status].name()));
    }

    if ((onroad != onroad_prev) || sm.frame < 2) {
      Hardware::set_brightness(50);
      Hardware::set_display_power(onroad);
    }
  });
  timer.start(50);

  return a.exec();
}
