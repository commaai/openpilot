#pragma once

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/ui.h"

class DriverMonitoring {

public:
  explicit DriverMonitoring();
  void updateState(const UIState &s);
  void updateDmonitoring(UIState *s, const cereal::DriverStateV2::Reader &driverstate);
  void drawDriverState(QPainter &painter, int width, int height, const UIState *s);

private:
  QPixmap dm_img;
  bool dmActive = false;
  bool rightHandDM = false;
  float dm_fade_state = 1.0;
};
