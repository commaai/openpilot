#include <zmq.h>

#include <cstring>

#include "catch2/catch.hpp"
#include "json11.hpp"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

void log_thread(void *zctx, int tid, int msg_cnt) {
  for (int i = 0; i < msg_cnt; ++i) {
    LOGD("%d", tid);
    usleep(1);
  }
}

void recv_thread(void *zctx, int *thread_msg, size_t size) {
  void *sock = zmq_socket(zctx, ZMQ_PULL);
  zmq_bind(sock, "ipc:///tmp/logmessage");
  while (true) {
    char buf[2048] = {};
    zmq_recv(sock, buf, sizeof(buf), 0);
    if (strcmp("exit", buf) == 0) break;
    std::string err;
    auto msg = json11::Json::parse(buf, err);
    REQUIRE(!msg.is_null());
    int thread_id = atoi(msg["msg"].string_value().c_str());
    REQUIRE((thread_id >= 0 && thread_id < size));
    thread_msg[thread_id]++;
  }
  zmq_close(sock);
}

TEST_CASE("swaglog") {
  SECTION("multiple threads logging") {
    auto msg_cnt = GENERATE(1000, 3000, 6000);
    void *zctx = zmq_ctx_new();
    const int thread_cnt = 10;
    int thread_msg[thread_cnt] = {};

    // start recv thread
    std::thread rcv = std::thread(recv_thread, zctx, thread_msg, std::size(thread_msg));
    util::sleep_for(20);
    // start send thread
    std::vector<std::thread> send_threads;
    for (int tid = 0; tid < thread_cnt; ++tid) {
      send_threads.push_back(std::thread(log_thread, zctx, tid, msg_cnt));
    }
    for (auto &t : send_threads) t.join();

    // send exit mssg
    void *sock = zmq_socket(zctx, ZMQ_PUSH);
    zmq_connect(sock, "ipc:///tmp/logmessage");
    while (zmq_send(sock, "exit", 4, 0) < 0) {
      usleep(1);
    }
    zmq_close(sock);
    rcv.join();

    for (int i = 0; i < thread_cnt; ++i) {
      REQUIRE(thread_msg[i] == msg_cnt);
    }
    zmq_ctx_destroy(zctx);
  }

  
}
