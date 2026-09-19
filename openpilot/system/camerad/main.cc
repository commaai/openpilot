#include "system/camerad/cameras/camera_common.h"

#include <cassert>

#include "common/params.h"
#include "common/util.h"
#include "common/camera120.h"
#include "common/hardware/hw.h"

int main(int argc, char *argv[]) {
  if (camera120_enabled() && Hardware::get_device_type() != cereal::InitData::DeviceType::MICI) {
    fprintf(stderr, "CAMERA_720P60 requires comma four (MICI)\n");
    return 1;
  }
  // doesn't need RT priority since we're using isolcpus
  int ret = util::set_core_affinity({6});
  assert(ret == 0 || Params().getBool("IsOffroad")); // failure ok while offroad due to offlining cores

  camerad_thread();
  return 0;
}
