#pragma once

#include "selfdrive/hardware/base.h"

class Hardware : public HardwareBase {
public:
  Hardware() : HardwareBase() {}
  bool PC() override { return true; }
  std::string get_os_version() override { return "openpilot for PC"; }
};
