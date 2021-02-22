#pragma once

#include <string>

#include "panda.h"

std::string get_firmware_fn();
// Ensures panda is running firmware or throws an exception if it fails
void update_panda(std::string serial="", std::string dfu_serial="");

void build_st(std::string target, bool clean=true, bool output=false);
