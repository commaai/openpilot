
#include <sys/resource.h>

#include "common/ratekeeper.h"
#include "common/util.h"
#include "system/proclogd/proclog.h"

ExitHandler do_exit;

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  RateKeeper rk("proclogd", 0.5);
  PubMaster publisher({"procLog"});

  while (!do_exit) {
    MessageBuilder msg;
    buildProcLogMessage(msg);
    publisher.send("procLog", msg);

    rk.keepTime();
  }

  return 0;
}
