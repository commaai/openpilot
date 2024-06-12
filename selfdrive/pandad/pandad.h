#pragma once

#include <string>
#include <vector>

#include "selfdrive/pandad/panda.h"

bool safety_setter_thread(std::vector<Panda *> pandas);
void pandad_main_thread(std::vector<std::string> serials);
