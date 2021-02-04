#include <android/log.h>
#include <log/logger.h>
#include <log/logprint.h>

#include "common/util.h"
#include "messaging.hpp"

#define LOG_ID_KERNEL (log_id_t)5

int main() {
  ExitHandler do_exit;
  log_time last_log_time = {};

  PubMaster pm({"androidLog"});
  while (!do_exit) {
    // setup android logging
    logger_list *loggers = last_log_time.tv_sec == 0 ?
                           android_logger_list_alloc(ANDROID_LOG_RDONLY | ANDROID_LOG_NONBLOCK, 0, 0) :
                           android_logger_list_alloc_time(ANDROID_LOG_RDONLY | ANDROID_LOG_NONBLOCK, last_log_time, 0);
    assert(loggers);

    const log_id_t log_ids[] = {LOG_ID_MAIN, LOG_ID_RADIO, LOG_ID_SYSTEM, LOG_ID_CRASH, LOG_ID_KERNEL};
    for (const auto &id : log_ids) {
      struct logger *log = android_logger_open(loggers, id);
      assert(log != nullptr);
    }

    while (!do_exit) {
      log_msg log_msg;
      int err = android_logger_list_read(loggers, &log_msg);
      if (err <= 0) break;

      AndroidLogEntry entry;
      err = android_log_processLogBuffer(&log_msg.entry_v1, &entry);
      if (err == 0) {
        last_log_time.tv_sec = entry.tv_sec;
        last_log_time.tv_nsec = entry.tv_nsec;

        MessageBuilder msg;
        auto androidEntry = msg.initEvent().initAndroidLog();
        androidEntry.setId(log_msg.id());
        androidEntry.setTs(entry.tv_sec * 1000000000ULL + entry.tv_nsec);
        androidEntry.setPriority(entry.priority);
        androidEntry.setPid(entry.pid);
        androidEntry.setTid(entry.tid);
        androidEntry.setTag(entry.tag);
        androidEntry.setMessage(entry.message);

        pm.send("androidLog", msg);
      }
    }

    android_logger_list_free(loggers);
    util::sleep_for(500);
  }
  return 0;
}
