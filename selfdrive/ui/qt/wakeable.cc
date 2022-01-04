#include "selfdrive/ui/qt/wakeable.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/common/swaglog.h"

void Wakeable::setAwake(bool on) {
  if (on != awake) {
    awake = on;
    Hardware::set_display_power(awake);
    LOGD("setting display power %d", awake);
    emitDisplayPowerChanged(awake);
  }
}