#include <assert.h>
#include <string>
#include "common/swaglog.h"
#include "logger.h"
#include "messaging.hpp"

static kj::Array<capnp::word> build_boot_log() {
  MessageBuilder msg;
  auto boot = msg.initEvent().initBoot();

  boot.setWallTimeNanos(nanos_since_epoch());

  std::string lastKmsg = util::read_file("/sys/fs/pstore/console-ramoops");
  boot.setLastKmsg(capnp::Data::Reader((const kj::byte*)lastKmsg.data(), lastKmsg.size()));

  std::string lastPmsg = util::read_file("/sys/fs/pstore/pmsg-ramoops-0");
  boot.setLastPmsg(capnp::Data::Reader((const kj::byte*)lastPmsg.data(), lastPmsg.size()));

  std::string launchLog = util::read_file("/tmp/launch_log");
  boot.setLaunchLog(capnp::Text::Reader(launchLog.data(), launchLog.size()));
  return capnp::messageToFlatArray(msg);
}

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

  BZFile bz_file(path.c_str());

  // Write initdata
  bz_file.write(logger_build_init_data().asBytes());

  // Write bootlog
  bz_file.write(build_boot_log().asBytes());

  return 0;
}
