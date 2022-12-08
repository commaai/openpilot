#include <cassert>
#include <string>

#include "cereal/messaging/messaging.h"
#include "common/swaglog.h"
#include "selfdrive/loggerd/logger.h"


static kj::Array<capnp::word> build_boot_log() {
  MessageBuilder msg;
  auto boot = msg.initEvent().initBoot();

  boot.setWallTimeNanos(nanos_since_epoch());

  std::string pstore = "/sys/fs/pstore";
  std::map<std::string, std::string> pstore_map = util::read_files_in_dir(pstore);

  int i = 0;
  auto lpstore = boot.initPstore().initEntries(pstore_map.size());
  for (auto& kv : pstore_map) {
    auto lentry = lpstore[i];
    lentry.setKey(kv.first);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)kv.second.data(), kv.second.size()));
    i++;
  }

  // Gather output of commands
  std::vector<std::string> bootlog_commands = {
    "[ -x \"$(command -v journalctl)\" ] && journalctl",
  };

  if (Hardware::TICI()) {
    bootlog_commands.push_back("[ -e /dev/nvme0 ] && sudo nvme smart-log --output-format=json /dev/nvme0");
  }

  auto commands = boot.initCommands().initEntries(bootlog_commands.size());
  for (int j = 0; j < bootlog_commands.size(); j++) {
    auto lentry = commands[j];

    lentry.setKey(bootlog_commands[j]);

    const std::string result = util::check_output(bootlog_commands[j]);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)result.data(), result.size()));
  }

  boot.setLaunchLog(util::read_file("/tmp/launch_log"));
  return capnp::messageToFlatArray(msg);
}

int main(int argc, char** argv) {
  const std::string path = LOG_ROOT + "/boot/" + logger_get_route_name();
  LOGW("bootlog to %s", path.c_str());

  // Open bootlog
  bool r = util::create_directories(LOG_ROOT + "/boot/", 0775);
  assert(r);

  RawFile file(path.c_str());
  // Write initdata
  file.write(logger_build_init_data().asBytes());
  // Write bootlog
  file.write(build_boot_log().asBytes());

  return 0;
}
