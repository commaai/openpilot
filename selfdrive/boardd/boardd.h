#pragma once

#include "selfdrive/boardd/panda.h"

Panda *usb_connect(std::string serial = {}, uint32_t index = 0);
bool safety_setter_thread(std::vector<Panda *> pandas);
void boardd_main_thread(std::vector<std::string> serials);
