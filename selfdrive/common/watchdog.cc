#include "selfdrive/common/watchdog.h"

#include <charconv>
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

const std::string watchdog_fn_prefix = "/dev/shm/wd_";  // + <pid>

bool watchdog_kick() {
  static std::string fn = watchdog_fn_prefix + std::to_string(getpid());

  char str[64];
  auto [ptr, ec] = std::to_chars(str, str + std::size(str), nanos_since_boot());
  if (ec == std::errc()) {
    return util::write_file(fn.c_str(), str, ptr - str, O_WRONLY | O_CREAT) > 0;
  }
  return false;
}
