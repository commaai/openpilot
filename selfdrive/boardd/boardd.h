#pragma once

#include <string>
#include <vector>

#include "selfdrive/boardd/panda.h"

bool safety_setter_thread(std::vector<Panda *> pandas);
void boardd_main_thread(std::vector<std::string> serials);
