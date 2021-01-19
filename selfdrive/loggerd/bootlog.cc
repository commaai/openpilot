#include <assert.h>
#include <string>

#include "common/swaglog.h"
#include "common/util.h"
#include "logger.h"
#include "messaging.hpp"

int main(int argc, char** argv) {
  LoggerState logger;
  logger_init(&logger, "bootlog", false);

  char segment_path[4096];
  int err = logger_next(&logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), nullptr);
  assert(err == 0);
  LOGW("bootlog to %s", segment_path);

  MessageBuilder msg;
  auto boot = msg.initEvent().initBoot();

  boot.setWallTimeNanos(nanos_since_epoch());

  std::string lastKmsg = util::read_file("/sys/fs/pstore/console-ramoops");
  boot.setLastKmsg(capnp::Data::Reader((const kj::byte*)lastKmsg.data(), lastKmsg.size()));

  std::string lastPmsg = util::read_file("/sys/fs/pstore/pmsg-ramoops-0");
  boot.setLastPmsg(capnp::Data::Reader((const kj::byte*)lastPmsg.data(), lastPmsg.size()));

  std::string launchLog = util::read_file("/tmp/launch_log");
  boot.setLaunchLog(capnp::Text::Reader(launchLog.data(), launchLog.size()));

  auto bytes = msg.toBytes();
  logger_log(&logger, bytes.begin(), bytes.size(), false);

  logger_close(&logger);
  return 0;
}
