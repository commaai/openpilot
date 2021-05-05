#include <assert.h>
#include <string>
#include "common/swaglog.h"
#include "logger.h"
#include "messaging.h"

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

  const std::string path = LOG_ROOT + "/boot/" + logger_get_route_name() + ".bz2";
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
