#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <ctime>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <thread>
#include <atomic>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"

#include "selfdrive/common/util.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "cereal/messaging/messaging.h"

#include "panda.h"
#include "panda_dfu.h"
#include "panda_flashing.h"
#include "pigeon.h"

#define NIBBLE_TO_HEX(n) ((n) < 10 ? (n) + '0' : ((n) - 10) + 'A')
#define REQUEST_IN 192

std::string convert(unsigned short s){
  short left = s / 256;
  short right = s % 256;
  std::string ans = "";
  ans += NIBBLE_TO_HEX(right / 16);
  ans += NIBBLE_TO_HEX(right % 16);
  ans += NIBBLE_TO_HEX(left / 16);
  ans += NIBBLE_TO_HEX(left % 16);
  return ans;
}

std::string dfu_serial_from_serial(std::string serial){
  if (serial==""){
    return "";
  }
  std::vector<unsigned short> hex_values;
  for(int i = 0 ; i < serial.size(); i+=4){
    hex_values.push_back(stoi(serial.substr(i, 4), 0, 16));
  }
  assert(hex_values.size() == 6);
  unsigned short left = hex_values[1] + hex_values[5];
  unsigned short middle = hex_values[0] + hex_values[4] + 0x0a00;
  unsigned short right = hex_values[3];
  //Split to bytes, cast to string
  return convert(left)+convert(middle)+convert(right);
}

const std::string basedir = util::getenv_default("BASEDIR", "", "/data/pythonpath");

void build_st(std::string target, bool clean, bool output) {
  std::string cmd = "cd " + basedir + "/panda/board";
  if (clean) cmd += " && make clean";
  cmd += " && make " + target;
  if (!output) cmd += " > /dev/null";

  system(cmd.c_str());
}

std::string get_firmware_fn() {
  std::string signed_fn = basedir + "/panda/board/obj/panda.bin.signed";

  if (util::file_exists(signed_fn)) {
    LOGW("Using prebuilt signed firmware");
    return signed_fn;
  } else {
    LOGW("Building panda firmware");
    build_st("obj/panda.bin");
    return basedir + "/panda/board/obj/panda.bin";
  }
}

std::string get_expected_signature() {
  std::string firmware_filename = get_firmware_fn();
  // TODO: check that file exists and has a length > 128
  std::string firmware_contents = util::read_file(firmware_filename);
  std::string firmware_sig = firmware_contents.substr(firmware_contents.length()-128);
  if (firmware_sig.length() != 128) {
    LOGE("Error getting the firmware signature");
  }
  return firmware_sig;
}


DynamicPanda::DynamicPanda(std::string serial, std::string dfu_serial) : serial(serial), dfu_serial(dfu_serial) {
  reconnect();
}

bool DynamicPanda::connect() {
  try{
    PandaComm(0x0483, 0xdf11, serial);
    LOGD("Found panda in DFU mode");
    pandaExists = false;
    bootstub = false;
    return true;
  } catch (std::runtime_error &e){}

  try{
    c = new PandaComm(0xbbaa, 0xddcc, serial);
    LOGD("Found panda in a good state");
    pandaExists = true;
    bootstub = false;
    return true;
  } catch (std::runtime_error &e){}

  try{
    c = new PandaComm(0xbbaa, 0xddee, serial);
    LOGD("Found panda in bootstub mode");
    pandaExists = true;
    bootstub = true;
    return true;
  } catch (std::runtime_error &e){}

  LOGW("Dynamic panda cannot find any non DFU panda");
  pandaExists = false;
  return false;
}

std::string DynamicPanda::get_version() {
  std::vector<uint8_t> fw_sig_buf(0x40);
  c->control_read(REQUEST_IN, 0xd6, 0, 0, &fw_sig_buf[0], 0x40);
  return std::string(fw_sig_buf.begin(), fw_sig_buf.end());
}

std::string DynamicPanda::get_signature() {
  std::vector<uint8_t> fw_sig_buf(128);
  c->control_read(REQUEST_IN, 0xd3, 0, 0, &fw_sig_buf[0], 64);
  c->control_read(REQUEST_IN, 0xd4, 0, 0, &fw_sig_buf[64], 64);
  return std::string(fw_sig_buf.begin(), fw_sig_buf.end());
}

void DynamicPanda::flash(std::string fw_fn) {
  LOGD(("Firmware string: " + fw_fn).c_str());
  LOGD(("flash: main version:" + get_version()).c_str());
  if (!bootstub) {
    reset(true, false);
  }
  assert(bootstub);

  std::string code = util::read_file(fw_fn);
  unsigned char code_data[code.length()];
  for (int i = 0 ; i < code.length() ; i++) {
    code_data[i] = code[i];
  }
  // confirm flashed is present
  std::vector<uint8_t> buf(12);
  c->control_read(REQUEST_IN, 0xb0, 0, 0, &buf[0], 12);
  assert(buf[4] == 0xde && buf[5] == 0xad && buf[6] == 0xd0 && buf[7] == 0x0d);

  //unlock flash
  LOGD("flash: unlocking");
  c->control_write(REQUEST_IN, 0xb1, 0, 0, nullptr, 0);

  // erase sectors 1-3
  LOGD("flash: erasing");
  for (int i = 1 ; i < 4 ; i++) {
    c->control_write(REQUEST_IN, 0xb2, i, 0, nullptr, 0);
  }

  // flash over EP2
  int STEP = 0x10;
  LOGD("flash: flashing");
  for(int i = 0 ; i < code.length() ; i += STEP) {
    c->usb_bulk_write(2, code_data + i, STEP);
  }
  //reset
  LOGD("flash: resetting");
  try {
    c->control_write(REQUEST_IN, 0xd8, 0, 0, nullptr, 0);
  } catch (std::runtime_error &e) {}

  reconnect();
}

bool DynamicPanda::reconnect() {
  util::sleep_for(1000);
  for (int i = 0 ; i < 15 ; i++) {
    if (connect()) {
      return true;
    } else {
      LOGD("reconnecting, trying for DFU panda");
      try {
        PandaDFU dfu(dfu_serial);
        LOGD("Found DFU panda, recovering");
        dfu.recover();
      } catch(std::runtime_error &e) {}
    }
    util::sleep_for(1000);
  }
  return false;
}

void DynamicPanda::reset(bool enter_bootstub, bool enter_bootloader) {
  try {
    if (enter_bootloader) {
      c->control_write(REQUEST_IN, 0xd1, 0, 0, nullptr, 0);
    } else if (enter_bootstub) {
      c->control_write(REQUEST_IN, 0xd1, 1, 0, nullptr, 0);
    } else {
      c->control_write(REQUEST_IN, 0xd8, 0, 0, nullptr, 0);
    }
  } catch(std::runtime_error &e) {}

  if (!enter_bootloader) {
    reconnect();
  }
}

void DynamicPanda::recover() {
  reset(true, false);
  reset(false, true);
  while(true) {
    LOGD("Waiting for DFU");
    util::sleep_for(100);
    try {
      PandaDFU dfu(dfu_serial);
      dfu.recover();
      break;
    } catch(std::runtime_error &e) {}
  }

  do {
    connect();
    LOGD("Looking for non DFU panda");
    util::sleep_for(100);
  } while(!pandaExists);

  flash(get_firmware_fn());
}

DynamicPanda::~DynamicPanda() {
  delete(c);
}


void get_out_of_dfu(std::string dfu_serial) {
  try {
    PandaDFU dfuPanda(dfu_serial);
    LOGD("Found DFU panda, running recovery");
    dfuPanda.recover();
  } catch(std::runtime_error &e){
    LOGD("DFU panda not found");
    return;
  }
}

bool update_panda(std::string serial) {
  LOGD("updating panda");
  LOGD("\n1: Move out of DFU\n");
  std::string dfu_serial = dfu_serial_from_serial(serial);
  get_out_of_dfu(dfu_serial);
  LOGD("\n2: Start DynamicPanda and run the required steps\n");

  std::string fw_fn = get_firmware_fn();
  std::string fw_signature = get_expected_signature();
 
  DynamicPanda tempPanda(serial, dfu_serial);

  std::string panda_signature = tempPanda.bootstub ? "": tempPanda.get_signature();
  LOGD(("fw_sig::panda_sig \n" + fw_signature + "\n" + panda_signature).c_str());

  if (tempPanda.bootstub || panda_signature != fw_signature) {
    LOGW("Panda firmware out of date, update required");
    tempPanda.flash(fw_fn);
    LOGD("Done flashing new firmware");
  }

  if (tempPanda.bootstub) {
    std::string bootstub_version = tempPanda.get_version();
    LOGW(("Flashed firmware not booting, flashing development bootloader. Bootstub verstion: " + bootstub_version).c_str());
    tempPanda.recover();
    LOGD("Done flashing dev bootloader and firmware");
  }

  if (tempPanda.bootstub) {
    LOGW("Panda still not booting, exiting");
    return false;
  }

  panda_signature = tempPanda.get_signature();
  if (panda_signature != fw_signature) {
    LOGW("Version mismatch after flashing, exiting");
    return false;
  }
  return true;
}

