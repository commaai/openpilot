#include <cassert>
#include <string>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/loggerd/logger.h"

static kj::Array<capnp::word> build_boot_log() {
  MessageBuilder msg;
  auto boot = msg.initEvent().initBoot();

  boot.setWallTimeNanos(nanos_since_epoch());

  std::string pstore = "/sys/fs/pstore";
  std::map<std::string, std::string> pstore_map;
  util::read_files_in_dir(pstore, &pstore_map);

  const std::vector<std::string> log_keywords = {"Kernel panic"};
  auto lpstore = boot.initPstore().initEntries(pstore_map.size());
  int i = 0;
  for (auto& kv : pstore_map) {
    auto lentry = lpstore[i];
    lentry.setKey(kv.first);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)kv.second.data(), kv.second.size()));
    i++;

    for (auto &k : log_keywords) {
      if (kv.second.find(k) != std::string::npos) {
        LOGE("%s: found '%s'", kv.first.c_str(), k.c_str());
      }
    }
  }

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
