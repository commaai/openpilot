#include <future>
#include <cassert>
#include <iostream>
#include <cstring>
#include <thread>

#include <nanomsg/nn.h>
#include <nanomsg/pubsub.h>
#include <nanomsg/tcp.h>

#define N 1024

int sub_sock(const char *endpoint) {
  int sock = nn_socket(AF_SP, NN_SUB);
  assert(sock >= 0);

  nn_setsockopt(sock, NN_SUB, NN_SUB_SUBSCRIBE,  "", 0);
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
  auto p_sock = pub_sock("tcp://*:10011");
  auto s_sock = sub_sock("tcp://127.0.0.1:10010");
  std::cout << "Ready!" << std::endl;

  char * msg = new char[N];

  while (true){
    int bytes = nn_recv(s_sock, msg, N, 0);
    nn_send(p_sock, msg, bytes, 0);
  }

  return 0;
}
