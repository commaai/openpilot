#include <systemd/sd-journal.h>

#include <cassert>
#include <csignal>
#include <map>
#include <string>

#include "json11.hpp"

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

ExitHandler do_exit;
int main(int argc, char *argv[]) {

  PubMaster pm({"androidLog"});

  sd_journal *journal;
  int err = sd_journal_open(&journal, 0);
  assert(err >= 0);
  err = sd_journal_get_fd(journal); // needed so sd_journal_wait() works properly if files rotate
  assert(err >= 0);
  err = sd_journal_seek_tail(journal);
  assert(err >= 0);

  // workaround for bug https://github.com/systemd/systemd/issues/9934
  // call sd_journal_previous_skip after sd_journal_seek_tail (like journalctl -f does) to makes things work.
  sd_journal_previous_skip(journal, 1);

  while (!do_exit) {
    err = sd_journal_next(journal);
    assert(err >= 0);

    // Wait for new message if we didn't receive anything
    if (err == 0) {
      err = sd_journal_wait(journal, 1000 * 1000);
      assert (err >= 0);
      continue; // Try again
    }

    uint64_t timestamp = 0;
    err = sd_journal_get_realtime_usec(journal, &timestamp);
    assert(err >= 0);

    const void *data;
    size_t length;
    std::map<std::string, std::string> kv;

    SD_JOURNAL_FOREACH_DATA(journal, data, length) {
      std::string str((char*)data, length);

      // Split "KEY=VALUE"" on "=" and put in map
      std::size_t found = str.find("=");
      if (found != std::string::npos) {
        kv[str.substr(0, found)] = str.substr(found + 1, std::string::npos);
      }
    }

    MessageBuilder msg;

    // Build message
    auto androidEntry = msg.initEvent().initAndroidLog();
    androidEntry.setTs(timestamp);
    androidEntry.setMessage(json11::Json(kv).dump());
    if (kv.count("_PID")) androidEntry.setPid(std::atoi(kv["_PID"].c_str()));
    if (kv.count("PRIORITY")) androidEntry.setPriority(std::atoi(kv["PRIORITY"].c_str()));
    if (kv.count("SYSLOG_IDENTIFIER")) androidEntry.setTag(kv["SYSLOG_IDENTIFIER"]);

    pm.send("androidLog", msg);
  }

  sd_journal_close(journal);
  return 0;
}
