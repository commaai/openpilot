#include <cassert>

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"

ExitHandler do_exit;

int main(int argc, char *argv[]) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_realtime_priority(53);
    assert(ret == 0);
    ret = util::set_core_affinity({Hardware::EON() ? 2 : 6});
    assert(ret == 0 || Params().getBool("IsOffroad"));  // failure ok while offroad due to offlining cores
  }

  start_camera_server();
}
