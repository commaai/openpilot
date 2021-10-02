#include "selfdrive/boardd/panda_dfu.h"

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

#define DFU_DNLOAD 1
#define DFU_UPLOAD 2
#define DFU_GETSTATUS 3
#define DFU_CLRSTATUS 4
#define DFU_ABORT 6

const std::string basedir = util::getenv("BASEDIR", "/data/pythonpath");

PandaDFU::PandaDFU(std::string dfu_serial) : PandaComm(0x0483, 0xdf11, dfu_serial) {}

std::array<uin8_t, 6> PandaDFU::get_status() {
  std::array<uin8_t, 6> stat = {};
  control_read(0x21, DFU_GETSTATUS, 0, 0, &stat[0], 6);
  return stat;
}

void PandaDFU::status() {
  while (get_status()[1] != 0)
    ;
}

void PandaDFU::clear_status() {
  auto stat = get_status();
  if (stat[4] == 0xa) {
    control_read(0x21, DFU_CLRSTATUS, 0, 0, nullptr, 0);
  } else if (stat[4] == 0x9) {
    control_write(0x21, DFU_ABORT, 0, 0, nullptr, 0);
    status();
  }
  stat = get_status();
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
  for (int i = 0; i < program.length(); i++) {
    padded_program[i] = program[i];
  }
  for (int i = 0; i < paddedLength / blockSize; i++) {
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
    get_status();
  } catch (std::runtime_error &e) {
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
