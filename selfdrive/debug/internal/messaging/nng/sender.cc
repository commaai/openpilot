#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>

#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>


#define N 1024
#define MSGS 1e5

nng_socket sub_sock(const char *endpoint) {
  nng_socket sock;
  int r;
  r = nng_sub0_open(&sock);
  assert(r == 0);

  nng_setopt(sock, NNG_OPT_SUB_SUBSCRIBE, "", 0);
  nng_setopt_ms(sock, NNG_OPT_RECVTIMEO, 100);

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
  // auto p_sock = pub_sock("tcp://*:10003");
  // auto s_sock = sub_sock("tcp://127.0.0.1:10004");

  auto p_sock = pub_sock("ipc:///tmp/feeds/3");
  auto s_sock = sub_sock("ipc:///tmp/feeds/2");


  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < MSGS; i++){
    nng_msg *msg;
    nng_msg_alloc(&msg, N);
    nng_sendmsg(p_sock, msg, 0);

    nng_msg *rmsg;
    int r = nng_recvmsg(s_sock, &rmsg, 0);

    if (r) {
      std::cout << "Timeout" << std::endl;
    }
  }
  auto end = std::chrono::steady_clock::now();


  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
  double throughput = ((double) MSGS / (double) elapsed);

  std::cout << "Elapsed: " << elapsed << " s" << std::endl;
  std::cout << "Throughput: " << throughput << " msg/s" << std::endl;

  return 0;
}
