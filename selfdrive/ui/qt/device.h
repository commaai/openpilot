#pragma once

#include <QObject>

#include "selfdrive/ui/ui.h"

#define BACKLIGHT_DT 0.05
#define BACKLIGHT_TS 10.00

const int BACKLIGHT_OFFROAD = 50;

// device management class
class Device : public QObject {
  Q_OBJECT

public:
  Device(QObject *parent = 0);
  bool isAwake() { return awake; }
  void setOffroadBrightness(int brightness) {
    offroad_brightness = std::clamp(brightness, 0, 100);
  }

private:
  bool awake = false;
  int interactive_timeout = 0;
  bool ignition_on = false;

  int offroad_brightness = BACKLIGHT_OFFROAD;
  int last_brightness = 0;
  FirstOrderFilter brightness_filter;
  QFuture<void> brightness_future;

  void updateBrightness(const UIState &s);
  void updateWakefulness(const UIState &s);
  void setAwake(bool on);

  signals:
    void displayPowerChanged(bool on);
  void interactiveTimeout();

  public slots:
    void resetInteractiveTimeout(int timeout = -1);
  void update(const UIState &s);
};

Device *device();
