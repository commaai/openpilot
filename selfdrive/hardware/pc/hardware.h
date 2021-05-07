#pragma once

#include <cstdlib>

#include "selfdrive/hardware/base.h"

class Hardware : public HardwareBase {
 public:
  Hardware() {}
  std::string get_os_version() const override { return "openpilot for PC"; }
  bool PC() const override { return true; }
};
