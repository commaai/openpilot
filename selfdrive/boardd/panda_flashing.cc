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

#include "common/util.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "messaging.hpp"

#include "panda.h"
#include "panda_flashing.hpp"
#include "pigeon.h"

#define DFU_DNLOAD 1
#define DFU_UPLOAD 2
#define DFU_GETSTATUS 3
#define DFU_CLRSTATUS 4
#define DFU_ABORT 6

const std::string basedir = util::getenv_default("BASEDIR", "", "/data/pythonpath");

void build_st(std::string target, bool clean=true, bool output=false) {
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


void PandaDFU::status() {
  std::vector<uint8_t> stat(6);
  while (true) {
    control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
    if (stat[1] == 0) {
      break;
    }
  }
}

void PandaDFU::clear_status() {
  std::vector<uint8_t> stat(6);
  control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
  if (stat[4] == 0xa) {
    control_read(0x21, DFU_CLRSTATUS, 0, 0, nullptr, 0);
  } else if(stat[4] == 0x9) {
    control_write(0x21, DFU_ABORT, 0, 0, nullptr, 0);
    status();
  }
  control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
}

void PandaDFU::erase(int adress) {
  unsigned char data[5];
  data[0] = 0x41;
  memcpy(&data[1], &adress, sizeof(adress));
  control_write(0x21, DFU_DNLOAD, 0, 0, data, 5);
  status();
}

void PandaDFU::program(int adress, std::string program) {
  int blockSize = 2048;

  // Set address pointer
  unsigned char data[5];
  data[0] = 0x21;
  memcpy(&data[1], &adress, sizeof(adress));
  control_write(0x21, DFU_DNLOAD, 0, 0, data, 5);
  status();

  // Program
  int paddedLength = program.length() + (blockSize - (program.length() % blockSize));
  unsigned char padded_program[paddedLength];
  std::fill(padded_program, padded_program + paddedLength, 0xff);
  // Not sure how to copy from string to unsigned char * with one command. For loop works
  for (int i = 0 ; i < program.length() ; i++) {
    padded_program[i] = program[i];
  }
  for (int i = 0 ; i < paddedLength/ blockSize ; i++) {
    LOGD("Programming with block %d", i);
    control_write(0x21, DFU_DNLOAD, 2 + i, 0, padded_program + blockSize * i, blockSize);
    status();
  }
  LOGD("Done with programming");
}

void PandaDFU::reset() {
  unsigned char data[5];
  data[0] = 0x21;
  int adress = 0x8000000;
  memcpy(&data[1], &adress, sizeof(adress));
  control_write(0x21, DFU_DNLOAD, 0, 0, data, 5);
  status();
  try {
    control_write(0x21, DFU_DNLOAD, 2, 0, nullptr, 0);
    unsigned char buf[6];
    control_read(0x21, DFU_GETSTATUS, 0, 0, buf, 6);
  } catch(std::runtime_error &e) {
    LOGE("DFU reset failed");
  }
}

void PandaDFU::program_bootstub(std::string program_file) {
   clear_status();
   erase(0x8004000);
   erase(0x8000000);
   program(0x8000000, program_file);
   reset();
}

void PandaDFU::recover() {
  build_st("obj/bootstub.panda.bin");
  std::string program = util::read_file(basedir + "/panda/board/obj/bootstub.panda.bin");
  program_bootstub(program);
}
PandaDFU::PandaDFU(std::string dfu_serial) : PandaComm(0x0483, 0xdf11, dfu_serial){}

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

void update_panda(std::string serial, std::string dfu_serial) {
  LOGD("updating panda");
  LOGD("\n1: Move out of DFU\n");
  get_out_of_dfu(dfu_serial);
  LOGD("\n2: Start DynamicPanda and run the required steps\n");

  std::string fw_fn = get_firmware_fn();
  std::string fw_signature = get_expected_signature();
  DynamicPanda tempPanda(serial, dfu_serial);
  std::string panda_signature = tempPanda.bootstub ? "": tempPanda.get_signature();
  LOGD("fw_sig::panda_sig");
  LOGD(fw_signature.c_str());
  LOGD(panda_signature.c_str());

  if (tempPanda.bootstub || panda_signature != fw_signature) {
    LOGW("Panda firmware out of date, update required");
    tempPanda.flash(fw_fn);
    LOGD("Done flashing new firmware");
  }

  if (tempPanda.bootstub) {
    std::string bootstub_version = tempPanda.get_version();
    LOGW("Flashed firmware not booting, flashing development bootloader. Bootstub verstion: ");
    LOGW(bootstub_version.c_str());
    tempPanda.recover();
    LOGD("Done flashing dev bootloader and firmware");
  }

  if (tempPanda.bootstub) {
    LOGW("Panda still not booting, exiting");
    throw std::runtime_error("PANDA NOT BOOTING");
  }

  panda_signature = tempPanda.get_signature();
  if (panda_signature != fw_signature) {
    LOGW("Version mismatch after flashing, exiting");
    throw std::runtime_error("FIRMWARE VERSION MISMATCH");
  }

}
