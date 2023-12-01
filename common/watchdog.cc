#include <string>

#include "common/watchdog.h"
#include "common/util.h"

const std::string watchdog_fn_prefix = "/dev/shm/wd_";  // + <pid>

bool watchdog_kick(uint64_t ts) {
  static std::string fn = watchdog_fn_prefix + std::to_string(getpid());
  return util::write_file(fn.c_str(), &ts, sizeof(ts), O_WRONLY | O_CREAT) > 0;
}
