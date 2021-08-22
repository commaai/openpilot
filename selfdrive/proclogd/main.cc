
#include <sys/resource.h>

#include "selfdrive/common/util.h"
#include "selfdrive/proclogd/proclog.h"

ExitHandler do_exit;

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  PubMaster publisher({"procLog"});
  while (!do_exit) {
    MessageBuilder msg;
    buildProcLogMessage(msg);
    publisher.send("procLog", msg);

    util::sleep_for(2000);  // 2 secs
  }

  return 0;
}
