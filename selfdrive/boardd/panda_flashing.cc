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
#include "pigeon.h"

#define DFU_DNLOAD 1
#define DFU_UPLOAD 2
#define DFU_GETSTATUS 3
#define DFU_CLRSTATUS 4
#define DFU_ABORT 6

std::string get_basedir(){ // Ends with /
  try {
    std::string basedir = getenv("BASEDIR");
    if (basedir == "") {
      LOGW("BASEDIR is not defined, provide the enviromental variable or run manager.py");
      return "";
    }
    return basedir;
  } catch (std::logic_error &e) {
    LOGW("BASEDIR is not defined, provide the enviromental variable or run manager.py");
    return "";
  }
}
void build_st(std::string target){
  system(("cd " + get_basedir() + "panda/board && make -f Makefile clean > /dev/null && make -f Makefile " + target +" > /dev/null ").c_str());
}

std::string get_firmware_fn(){
  std::string basedir = get_basedir();
  std::string signed_fn = basedir + "panda/board/obj/panda.bin.signed";

  if (util::file_exists(signed_fn)) {
    LOGW("Using prebuilt signed firmware");
    return signed_fn;
  } else {
    LOGW("Building panda firmware");
    build_st("obj/panda.bin");
    return basedir + "panda/board/obj/panda.bin";
  }
}

std::string get_expected_signature(){
  std::string firmware_filename = get_firmware_fn();
  std::string firmware_contents = util::read_file(firmware_filename);
  std::string firmware_sig = firmware_contents.substr(firmware_contents.length()-128);
  if (firmware_sig.length() != 128) {
    std::cout<<"Error getting the firmware signature"<<std::endl;
  }
  return firmware_sig;
}


void dfu_status(PandaComm* dfuPanda){
  std::vector<uint8_t> stat(6);
  while (true) {
    dfuPanda->control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
    if (stat[1] == 0) {
      break;
    }
  }
}

void dfu_clear_status(PandaComm* dfuPanda){
  std::vector<uint8_t> stat(6);
  dfuPanda->control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
  if (stat[4] == 0xa) {
    dfuPanda->control_read(0x21, DFU_CLRSTATUS, 0, 0, nullptr, 0);
  } else if(stat[4] == 0x9) {
    dfuPanda->control_write(0x21, DFU_ABORT, 0, 0, nullptr, 0);
    dfu_status(dfuPanda);
  }
  dfuPanda->control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
}

unsigned char* pack_int(int num){
  unsigned char* data = new unsigned char[4];
  memcpy(data, &num, sizeof(num));
  return data;
}

void dfu_erase(PandaComm* dfuPanda, int adress){
  unsigned char data[5];
  data[0] = 0x41;
  unsigned char* packed = pack_int(adress);
  for (int i = 0 ; i < 4 ; i++) {
    data[i+1] = packed[i];
  }
  dfuPanda->control_write(0x21, DFU_DNLOAD, 0, 0, data, 5);
  dfu_status(dfuPanda);
}

void dfu_program(PandaComm* dfuPanda, int adress, std::string program){
  int blockSize = 2048;

  // Set address pointer
  unsigned char data[5];
  data[0] = 0x21;
  unsigned char* packed = pack_int(adress);
  for (int i = 0 ; i < 4 ; i++) {
    data[i+1] = packed[i];
  }
  dfuPanda->control_write(0x21, DFU_DNLOAD, 0, 0, data, 5);
  dfu_status(dfuPanda);

  // Program
  int paddedLength = program.length() + (blockSize - (program.length() % blockSize));
  unsigned char padded_program[paddedLength];
  std::fill(padded_program, padded_program + paddedLength, 0xff);
  for (int i = 0 ; i < program.length() ; i++) {
    padded_program[i]=program[i];
  }
  for (int i = 0 ; i < paddedLength/ blockSize ; i++) {
    std::cout<<"Programming with block "<<i<<std::endl;
    dfuPanda->control_write(0x21, DFU_DNLOAD, 2 + i, 0, padded_program + blockSize * i, blockSize);
    dfu_status(dfuPanda);
  }
  std::cout<<"Done with programming"<<std::endl;
}

void dfu_reset(PandaComm* dfuPanda){
  unsigned char data[5];
  data[0] = 0x21;
  unsigned char* packed = pack_int(0x8000000);
  for (int i = 0 ; i < 4 ; i++) {
    data[i+1] = packed[i];
  }
  dfuPanda->control_write(0x21, DFU_DNLOAD, 0, 0, data, 5);
  dfu_status(dfuPanda);
  try {
    dfuPanda->control_write(0x21, DFU_DNLOAD, 2, 0, nullptr, 0);
    unsigned char buf[6];
    dfuPanda->control_read(0x21, DFU_GETSTATUS, 0, 0, buf, 6);
  } catch(std::runtime_error &e) {
    std::cout<<"DFU reset failed"<<std::endl;
  }
}

void dfu_program_bootstub(PandaComm* dfuPanda, std::string program){
   dfu_clear_status(dfuPanda);
   dfu_erase(dfuPanda, 0x8004000);
   dfu_erase(dfuPanda, 0x8000000);
   dfu_program(dfuPanda, 0x8000000, program);
   dfu_reset(dfuPanda);
}

void dfu_recover(PandaComm* dfuPanda){
  std::string basedir = get_basedir();
  build_st("obj/bootstub.panda.bin");
  std::string program = util::read_file(basedir+"panda/board/obj/bootstub.panda.bin");
  dfu_program_bootstub(dfuPanda, program);
}

void get_out_of_dfu(){
  PandaComm* dfuPanda;
  try {
    dfuPanda = new PandaComm(0x0483, 0xdf11);
  } catch(std::runtime_error &e) {
    std::cout<<"DFU panda not found"<<std::endl;
    delete(dfuPanda);
    return;
  }
  std::cout<<"Panda in DFU mode found, flashing recovery"<<std::endl;
  util::sleep_for(2500);
  dfu_recover(dfuPanda);
  delete(dfuPanda);
}

void update_panda(){
  std::cout<<"updating panda"<<std::endl;
  std::cout<<std::endl<<"1: Move out of DFU"<<std::endl<<std::endl;
  util::sleep_for(500);
  get_out_of_dfu();
  std::cout<<std::endl<<"2: Start DynamicPanda and run the required steps"<<std::endl<<std::endl;
  util::sleep_for(2500);
  
  std::string fw_fn = get_firmware_fn();
  std::string fw_signature = get_expected_signature();
  DynamicPanda tempPanda;
  util::sleep_for(1000);
  std::string panda_signature = tempPanda.bootstub ? "": tempPanda.get_signature();
  std::cout<<"fw_sig::panda_sig"<<std::endl<<fw_signature<<std::endl<<panda_signature<<std::endl;
  util::sleep_for(2500);

  if (tempPanda.bootstub || panda_signature != fw_signature) {
    std::cout<<std::endl<<"Panda firmware out of date, update required"<<std::endl;
    tempPanda.flash(fw_fn);
    std::cout<<std::endl<<"Done flashing new firmware"<<std::endl;
    util::sleep_for(2500);
  }

  if (tempPanda.bootstub) {
    std::string bootstub_version = tempPanda.get_version();
    std::cout<<"Flashed firmware not booting, flashing development bootloader. Bootstub verstion "<<bootstub_version<<std::endl;
    util::sleep_for(2500);
    tempPanda.recover();
    std::cout<<"Done flashing dev bootloader and firmware"<<std::endl;
    util::sleep_for(2500);
  }

  if (tempPanda.bootstub) {
    std::cout<<"Panda still not booting, exiting"<<std::endl;
    util::sleep_for(2500);
    throw std::runtime_error("PANDA NOT BOOTING");
  }

  panda_signature = tempPanda.get_signature();
  if (panda_signature != fw_signature) {
    std::cout<<"Version mismatch after flashing, exiting"<<std::endl;
    util::sleep_for(2500);
    throw std::runtime_error("FIRMWARE VERSION MISMATCH");
  }

} 
