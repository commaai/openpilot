#include <signal.h>      /* Definition of SIG* constants */
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>

#include <future>

#include "catch2/catch.hpp"
#include "common/params.h"
#include "common/timing.h"
#include "common/util.h"


std::atomic<int> signal_caught = 0;

void signal_handle(int s) {
  signal_caught = s;
}

TEST_CASE("params_blocking_read") {
  bool send_sigint = GENERATE(true, false);
  char tmp_path[] = "/tmp/test_Params_XXXXXX";
  const std::string param_path = mkdtemp(tmp_path);
  pthread_t thread_id = pthread_self();

  // install signal handle
  signal_caught = 0;
  std::signal(SIGINT, signal_handle);

  // write param in thread
  double write_after = 200;
  std::future<void> future = std::async(std::launch::async, [=]() {
    util::sleep_for(write_after);
    if (send_sigint) {
      pthread_kill(thread_id, SIGINT);
    }
    Params params(param_path);
    params.put("CarParams", "1");
  });

  // blocking read
  double begin_ts = millis_since_boot();
  Params params(param_path);
  auto value = params.get("CarParams", true);
  if (send_sigint) {
    REQUIRE(signal_caught == SIGINT);
    REQUIRE(value.empty());
  } else {
    REQUIRE(signal_caught == 0);
    REQUIRE(value == "1");
  }
  double end_ts = millis_since_boot();
  REQUIRE((end_ts - begin_ts) > write_after);
}
