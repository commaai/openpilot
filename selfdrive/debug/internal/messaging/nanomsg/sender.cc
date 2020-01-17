#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>

#include <nanomsg/nn.h>
#include <nanomsg/pubsub.h>
#include <nanomsg/tcp.h>


#define N 1024
#define MSGS 1e5

int sub_sock(const char *endpoint) {
  int sock = nn_socket(AF_SP, NN_SUB);
  assert(sock >= 0);

  nn_setsockopt(sock, NN_SUB, NN_SUB_SUBSCRIBE,  "", 0);

  int timeout = 100;
  nn_setsockopt(sock, NN_SOL_SOCKET, NN_RCVTIMEO, &timeout , sizeof(timeout));

  assert(nn_connect(sock, endpoint) >= 0);
  return sock;
}

int pub_sock(const char *endpoint){
  int sock = nn_socket(AF_SP, NN_PUB);
  assert(sock >= 0);

  int b = 1;
  nn_setsockopt(sock, NN_TCP, NN_TCP_NODELAY, &b, sizeof(b));

  assert(nn_bind(sock, endpoint) >= 0);

  return sock;
}


int main(int argc, char *argv[]) {
  auto p_sock = pub_sock("tcp://*:10010");
  auto s_sock = sub_sock("tcp://127.0.0.1:10011");
  std::cout << "Ready!" << std::endl;

  // auto p_sock = pub_sock("ipc:///tmp/feeds/3");
  // auto s_sock = sub_sock("ipc:///tmp/feeds/2");

  char * msg = new char[N];
  auto start = std::chrono::steady_clock::now();


  for (int i = 0; i < MSGS; i++){
    sprintf(msg, "%d", i);

    nn_send(p_sock, msg, N, 0);
    int bytes = nn_recv(s_sock, msg, N, 0);

    if (bytes < 0) {
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
