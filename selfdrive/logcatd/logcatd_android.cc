#include <android/log.h>
#include <log/logger.h>
#include <log/logprint.h>
#include <sys/resource.h>

#include <csignal>

#include "cereal/messaging/messaging.h"

#undef LOG_ID_KERNEL
#define LOG_ID_KERNEL 5

int main() {
  std::signal(SIGINT, exit);
  std::signal(SIGTERM, exit);
  setpriority(PRIO_PROCESS, 0, -15);

  // setup android logging
  logger_list *logger_list = android_logger_list_alloc(ANDROID_LOG_RDONLY, 0, 0);
  assert(logger_list);
  for (auto log_id : {LOG_ID_MAIN, LOG_ID_RADIO, LOG_ID_SYSTEM, LOG_ID_CRASH, (log_id_t)LOG_ID_KERNEL}) {
    logger *logger = android_logger_open(logger_list, log_id);
    assert(logger);
  }

  PubMaster pm({"androidLog"});

  while (true) {
    log_msg log_msg;
    int err = android_logger_list_read(logger_list, &log_msg);
    if (err <= 0) break;

    AndroidLogEntry entry;
    err = android_log_processLogBuffer(&log_msg.entry_v1, &entry);
    if (err < 0) continue;

    MessageBuilder msg;
    auto androidEntry = msg.initEvent().initAndroidLog();
    androidEntry.setId(log_msg.id());
    androidEntry.setTs(entry.tv_sec * NS_PER_SEC + entry.tv_nsec);
    androidEntry.setPriority(entry.priority);
    androidEntry.setPid(entry.pid);
    androidEntry.setTid(entry.tid);
    androidEntry.setTag(entry.tag);
    androidEntry.setMessage(entry.message);
    pm.send("androidLog", msg);
  }

  android_logger_list_free(logger_list);
  return 0;
}
