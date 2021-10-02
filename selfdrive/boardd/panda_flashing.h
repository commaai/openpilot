#pragma once

#include <string>

#include "selfdrive/boardd/panda.h"

std::string get_firmware_fn();
// Ensures panda is running firmware or throws an exception if it fails
bool update_panda(std::string serial = "");
void build_st(std::string target, bool clean = true, bool output = false);

// DynamicPanda class is used while setting up the "real" panda; aka the Panda that is running the firmware
// We need to be able to switch between different states before getting there though
class DynamicPanda {
 private:
  PandaComm* c;
  void cleanup();
  bool connect();
  bool reconnect();
  std::string serial;
  std::string dfu_serial;

 public:
  DynamicPanda(std::string serial, std::string dfu_serial);
  ~DynamicPanda();
  std::string get_version();
  std::string get_signature();
  void flash(std::string fw_fn);
  void reset(bool enter_bootstub, bool enter_bootloader);
  void recover();
  bool pandaExists;
  bool bootstub;
};
