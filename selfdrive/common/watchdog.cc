#include "selfdrive/common/watchdog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

const std::string watchdog_fn_prefix = "/dev/shm/wd_";  // + <pid>

bool watchdog_kick() {
  static std::string fn = watchdog_fn_prefix + std::to_string(getpid());

  uint64_t ts = nanos_since_boot();
  return util::write_file(fn.c_str(), &ts, sizeof(ts), O_WRONLY | O_CREAT) > 0;
}
