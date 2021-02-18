#include <string>

#include "panda.h"

std::string get_basedir();
std::string get_firmware_fn();
void build_st(std::string target);
void get_out_of_dfu(std::string dfu_serial="");
void dfu_recover(PandaComm* dfuPanda);
void update_panda(std::string serial="", std::string dfu_serial="");
