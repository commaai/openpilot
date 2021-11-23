#include "selfdrive/loggerd/loggerd.h"

#include <sys/resource.h>

int main(int argc, char** argv) {
  if (Hardware::EON()) {
    setpriority(PRIO_PROCESS, 0, -20);
  } else if (Hardware::TICI()) {
    int ret;
    ret = set_core_affinity({0, 1, 2, 3});
    assert(ret == 0);
    // TODO: why does this impact camerad timings?
    //ret = set_realtime_priority(1);
    //assert(ret == 0);
  }

  start_logging();

  return 0;
}
