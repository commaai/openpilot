#include "selfdrive/camerad/cameras/camera_common.h"

#include <cassert>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

int main(int argc, char *argv[]) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_realtime_priority(53);
    assert(ret == 0);
    ret = util::set_core_affinity({Hardware::EON() ? 2 : 6});
    assert(ret == 0 || Params().getBool("IsOffroad")); // failure ok while offroad due to offlining cores
  }

  camerad_thread();
  return 0;
}
