#include "selfdrive/loggerd/loggerd.h"

#include <sys/resource.h>

int main(int argc, char** argv) {
  if (Hardware::TICI()) {
    int ret;
    ret = util::set_core_affinity({0, 1, 2, 3});
    assert(ret == 0);
    // TODO: why does this impact camerad timings?
    //ret = util::set_realtime_priority(1);
    //assert(ret == 0);
  }

  loggerd_thread();

  return 0;
}
