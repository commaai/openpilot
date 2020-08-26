#include <iostream>
#include <cassert>
#include <string>
#include <map>

#include "json11.hpp"
#include <systemd/sd-journal.h>

#include "common/timing.h"
#include "messaging.hpp"

int main(int argc, char *argv[]) {
  int r;
  sd_journal *journal;

  PubMaster pm({"androidLog"});

  assert(sd_journal_open(&journal, 0) >= 0);

  // needed so sd_journal_wait() works properly if files rotate
  assert(sd_journal_get_fd(journal) >= 0);

  assert(sd_journal_seek_tail(journal) >= 0);

  while (true) {
    r = sd_journal_next(journal);

    // Wait for new message if we didn't receive anything
    if (r == 0){
      do {
        r = sd_journal_wait(journal, (uint64_t)-1);
        assert(r >= 0);
      } while (r == SD_JOURNAL_NOP);

      r = sd_journal_next(journal);
      if (r == 0) continue; // Try again if we still didn't get anything
    }

    uint64_t timestamp = 0;
    r = sd_journal_get_realtime_usec(journal, &timestamp);
    assert(r >= 0);

    const void *data;
    size_t length;
    std::map<std::string, std::string> kv;

    SD_JOURNAL_FOREACH_DATA(journal, data, length){
      std::string str((char*)data, length);

      std::size_t found = str.find("=");

      if (found!=std::string::npos){
        std::string key = str.substr(0, found);
        std::string value = str.substr(found + 1, std::string::npos);
        kv[key] = value;
      }

      json11::Json json(kv);

      capnp::MallocMessageBuilder msg;
      cereal::Event::Builder event = msg.initRoot<cereal::Event>();
      event.setLogMonoTime(nanos_since_boot());
      auto androidEntry = event.initAndroidLog();

      if (kv.count("_PID")){
        androidEntry.setPid(std::atoi(kv["_PID"].c_str()));
      }
      if (kv.count("PRIORITY")){
        androidEntry.setPriority(std::atoi(kv["PRIORITY"].c_str()));
      }
      if (kv.count("SYSLOG_IDENTIFIER")){
        androidEntry.setTag(kv["SYSLOG_IDENTIFIER"]);
      }

      androidEntry.setTs(timestamp);
      androidEntry.setMessage(json.dump());
      pm.send("androidLog", msg);
    }
  }

  sd_journal_close(journal);
  return 0;
}
