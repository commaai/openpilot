#include "system/camerad/cameras/camera_common.h"

#include <cassert>

#ifdef QCOM2
#include "CL/cl_ext_qcom.h"
#else
#define CL_PRIORITY_HINT_HIGH_QCOM NULL
#define CL_CONTEXT_PRIORITY_HINT_QCOM NULL
#endif
#include "common/clutil.h"
#include "common/params.h"
#include "common/util.h"

void camerad_thread(cl_device_id, cl_context);

int main(int argc, char *argv[]) {
  int ret = util::set_realtime_priority(53);
  assert(ret == 0);
  ret = util::set_core_affinity({6});
  assert(ret == 0 || Params().getBool("IsOffroad")); // failure ok while offroad due to offlining cores

  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
  cl_context ctx = CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err));

  camerad_thread(device_id, ctx);

  clReleaseContext(ctx);
  return 0;
}
