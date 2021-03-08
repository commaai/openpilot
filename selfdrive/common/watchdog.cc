#include <string>
#include <cstdint>
#include <unistd.h>

#include "common/timing.h"
#include "common/util.h"
#include "common/watchdog.h"

const std::string watchdog_fn_prefix = "/dev/shm/wd_";  // + <pid>

bool watchdog_kick(){
  std::string fn = watchdog_fn_prefix + std::to_string(getpid());
  std::string cur_t = std::to_string(nanos_since_boot());

  int r = write_file(fn.c_str(), cur_t.data(), cur_t.length(), O_WRONLY | O_CREAT);
  return r == 0;
}
