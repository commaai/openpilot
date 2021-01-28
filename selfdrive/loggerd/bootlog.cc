#include <assert.h>
#include <string>
#include <cstdio>

#include <bzlib.h>

#include "common/swaglog.h"
#include "common/util.h"
#include "messaging.hpp"
#include "logger.h"

int main(int argc, char** argv) {
  char filename[64] = {'\0'};

  time_t rawtime = time(NULL);
  struct tm timeinfo;

  localtime_r(&rawtime, &timeinfo);
  strftime(filename, sizeof(filename),
           "%Y-%m-%d--%H-%M-%S.bz2", &timeinfo);

  std::string path = LOG_ROOT + "/boot/" + std::string(filename);
  LOGW("bootlog to %s", path.c_str());

  // Open bootlog
  int r = logger_mkpath((char*)path.c_str());
  assert(r == 0);

  FILE * file = fopen(path.c_str(), "wb");
  assert(file != nullptr);

  // Open as bz2
  int bzerror;
  BZFILE* bz_file = BZ2_bzWriteOpen(&bzerror, file, 9, 0, 30);
  assert(bzerror == BZ_OK);

  // Write initdata
  kj::Array<capnp::word> init_msg = logger_build_init_data();
  auto bytes = init_msg.asBytes();
  BZ2_bzWrite(&bzerror, bz_file, bytes.begin(), bytes.size());
  assert(bzerror == BZ_OK);

  // Write bootlog
  kj::Array<capnp::word> boot_msg = logger_build_boot();
  bytes = boot_msg.asBytes();
  BZ2_bzWrite(&bzerror, bz_file, bytes.begin(), bytes.size());
  assert(bzerror == BZ_OK);

  // Close bz2 and file
  BZ2_bzWriteClose(&bzerror, bz_file, 0, NULL, NULL);
  assert(bzerror == BZ_OK);

  fclose(file);
  return 0;
}
