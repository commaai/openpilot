#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common/swaglog.h"

#include <cassert>
#include <mutex>
#include <string>

#include <stdarg.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "json11/json11.hpp"
#include "common/version.h"
#include "common/hardware/hw.h"

class SwaglogState {
public:
  SwaglogState() {
    sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    int sndbuf = SWAGLOG_MAX_SIZE;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, Path::swaglog_ipc().c_str(), sizeof(addr.sun_path) - 1);

    // workaround for https://github.com/dropbox/json11/issues/38
    setlocale(LC_NUMERIC, "C");

    print_level = CLOUDLOG_WARNING;
    if (const char* print_lvl = getenv("LOGPRINT")) {
      if (strcmp(print_lvl, "debug") == 0) {
        print_level = CLOUDLOG_DEBUG;
      } else if (strcmp(print_lvl, "info") == 0) {
        print_level = CLOUDLOG_INFO;
      } else if (strcmp(print_lvl, "warning") == 0) {
        print_level = CLOUDLOG_WARNING;
      }
    }

    ctx_j = json11::Json::object{};
    if (char* dongle_id = getenv("DONGLE_ID")) {
      ctx_j["dongle_id"] = dongle_id;
    }
    if (char* git_origin = getenv("GIT_ORIGIN")) {
      ctx_j["origin"] = git_origin;
    }
    if (char* git_branch = getenv("GIT_BRANCH")) {
      ctx_j["branch"] = git_branch;
    }
    if (char* git_commit = getenv("GIT_COMMIT")) {
      ctx_j["commit"] = git_commit;
    }
    if (char* daemon_name = getenv("MANAGER_DAEMON")) {
      ctx_j["daemon"] = daemon_name;
    }
    ctx_j["version"] = COMMA_VERSION;
    ctx_j["dirty"] = !getenv("CLEAN");
    ctx_j["device"] = Hardware::get_name();
  }

  ~SwaglogState() {
    close(sock);
  }

  void log(int levelnum, const char* filename, int lineno, const char* func, const char* msg, const std::string& log_s) {
    std::lock_guard lk(lock);
    if (levelnum >= print_level) {
      printf("%s: %s\n", filename, msg);
    }
    // drop if logmessaged is busy, not running, or the message is too big
    sendto(sock, log_s.data(), log_s.length(), MSG_DONTWAIT, (struct sockaddr*)&addr, sizeof(addr));
  }

  std::mutex lock;
  int sock = -1;
  struct sockaddr_un addr = {};
  int print_level;
  json11::Json::object ctx_j;
};

static void cloudlog_common(int levelnum, const char* filename, int lineno, const char* func, char* msg_buf) {
  static SwaglogState s;

  json11::Json::object log_j = json11::Json::object {
    {"ctx", s.ctx_j},
    {"levelnum", levelnum},
    {"filename", filename},
    {"lineno", lineno},
    {"funcname", func},
    {"created", seconds_since_epoch()},
    {"msg", msg_buf},
  };

  std::string log_s;
  log_s += (char)levelnum;
  ((json11::Json)log_j).dump(log_s);
  s.log(levelnum, filename, lineno, func, msg_buf, log_s);

  free(msg_buf);
}

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char* msg_buf = nullptr;
  int ret = vasprintf(&msg_buf, fmt, args);
  va_end(args);
  if (ret <= 0 || !msg_buf) return;
  cloudlog_common(levelnum, filename, lineno, func, msg_buf);
}
