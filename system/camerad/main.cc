#include "system/camerad/cameras/camera_common.h"

#include <cassert>
#include <cstdlib>

#include "common/params.h"
#include "common/util.h"

int main(int argc, char *argv[]) {
  // Check if we should use Jetson camera implementation
  bool use_jetson = (getenv("USE_JETSON_CAMERA") != nullptr) ||
                    (getenv("USE_V4L2_CAMERA") != nullptr);

  if (use_jetson) {
    // Jetson camera implementation - GStreamer based
    jetson_camerad_thread();
  } else {
    // Original QCOM/Spectra ISP path
    // doesn't need RT priority since we're using isolcpus
    int ret = util::set_core_affinity({6});
    assert(ret == 0 || Params().getBool("IsOffroad")); // failure ok while offroad due to offlining cores

    camerad_thread();
  }

  return 0;
}
