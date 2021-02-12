#pragma once
#include <string>

#include "panda.h"

std::string get_basedir();
std::string get_firmware_fn();
void build_st(std::string target);
void get_out_of_dfu();
void dfu_recover(PandaComm* dfuPanda);
