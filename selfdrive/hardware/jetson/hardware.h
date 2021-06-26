#pragma once

#include <cstdlib>

#include "selfdrive/common/util.h"
#include "selfdrive/hardware/base.h"

class HardwareJetson : public HardwareNone {
public:

  static bool JETSON() { return true; }

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
};
