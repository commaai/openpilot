#include <cassert>

#include "selfdrive/boardd/boardd.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "system/hardware/hw.h"

int main(int argc, char *argv[]) {
  LOGW("starting boardd");

  if (!Hardware::PC()) {
    int err;
    err = util::set_realtime_priority(54);
    assert(err == 0);
    err = util::set_core_affinity({4});
    assert(err == 0);
  }

  std::vector<std::string> serials(argv + 1, argv + argc);
  boardd_main_thread(serials);
  return 0;
}
