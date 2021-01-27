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

  MessageBuilder msg;
  auto boot = msg.initEvent().initBoot();

  boot.setWallTimeNanos(nanos_since_epoch());

  std::string lastKmsg = util::read_file("/sys/fs/pstore/console-ramoops");
  boot.setLastKmsg(capnp::Data::Reader((const kj::byte*)lastKmsg.data(), lastKmsg.size()));

  std::string lastPmsg = util::read_file("/sys/fs/pstore/pmsg-ramoops-0");
  boot.setLastPmsg(capnp::Data::Reader((const kj::byte*)lastPmsg.data(), lastPmsg.size()));

  std::string launchLog = util::read_file("/tmp/launch_log");
  boot.setLaunchLog(capnp::Text::Reader(launchLog.data(), launchLog.size()));


  // Open bootlog
  int r = logger_mkpath((char*)path.c_str());
  assert(r == 0);

  FILE * file = fopen(path.c_str(), "wb");
  assert(file != nullptr);

  // Open as bz2
  int bzerror;
  BZFILE* bz_file = BZ2_bzWriteOpen(&bzerror, file, 9, 0, 30);
  assert(bzerror == BZ_OK);

  // Write bootlog in bz2
  auto bytes = msg.toBytes();
  BZ2_bzWrite(&bzerror, bz_file, bytes.begin(), bytes.size());
  assert(bzerror == BZ_OK);

  // Close bz2 and file
  BZ2_bzWriteClose(&bzerror, bz_file, 0, NULL, NULL);
  assert(bzerror == BZ_OK);

  fclose(file);
  return 0;
}
