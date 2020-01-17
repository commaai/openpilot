#include <future>
#include <iostream>
#include <cstring>

#include <zmq.h>

// #define IPC

void *sub_sock(void *ctx, const char *endpoint) {
  void* sock = zmq_socket(ctx, ZMQ_SUB);
  zmq_connect(sock, endpoint);
  zmq_setsockopt(sock, ZMQ_SUBSCRIBE, "", 0);

  return sock;
}

void *pub_sock(void *ctx, const char *endpoint){
  void * sock = zmq_socket(ctx, ZMQ_PUB);

  zmq_bind(sock, endpoint);

  return sock;
}

int main(int argc, char *argv[]) {
  auto ctx = zmq_ctx_new();

#ifdef IPC
  auto s_sock = sub_sock(ctx, "ipc:///tmp/q0");
  auto p_sock = pub_sock(ctx, "ipc:///tmp/q1");
 #else
  auto s_sock = sub_sock(ctx, "tcp://localhost:10005");
  auto p_sock = pub_sock(ctx, "tcp://*:10004");
 #endif

  zmq_msg_t msg;
  zmq_msg_init(&msg);


  while (true){
    zmq_msg_recv(&msg, s_sock, 0);
    zmq_msg_send(&msg, p_sock, ZMQ_DONTWAIT);
  }

  zmq_msg_close(&msg);
  zmq_close(p_sock);
  zmq_close(s_sock);
  return 0;
}
