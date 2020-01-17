#include <future>
#include <cassert>
#include <iostream>
#include <cstring>
#include <thread>

#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

nng_socket sub_sock(const char *endpoint) {
  nng_socket sock;
  int r;
  r = nng_sub0_open(&sock);
  assert(r == 0);

  nng_setopt(sock, NNG_OPT_SUB_SUBSCRIBE, "", 0);

  while (true){
    r = nng_dial(sock, endpoint, NULL, 0);

    if (r == 0){
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  }
  return sock;
}

nng_socket pub_sock(const char *endpoint){
  nng_socket sock;
  int r;
  r = nng_pub0_open(&sock);
  assert(r == 0);
  r = nng_listen(sock, endpoint, NULL, 0);
  assert(r == 0);

  return sock;
}

int main(int argc, char *argv[]) {
  // auto p_sock = pub_sock("tcp://*:10004");
  // auto s_sock = sub_sock("tcp://127.0.0.1:10003");

  auto p_sock = pub_sock("ipc:///tmp/feeds/2");
  auto s_sock = sub_sock("ipc:///tmp/feeds/3");

  while (true){
    nng_msg *msg;
    nng_recvmsg(s_sock, &msg, 0);
    nng_sendmsg(p_sock, msg, 0);
  }

  return 0;
}
