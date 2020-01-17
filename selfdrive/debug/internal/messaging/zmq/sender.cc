#include <iostream>
#include <zmq.h>
#include <chrono>

#define N 1024
#define MSGS 1e5

// #define IPC

void *sub_sock(void *ctx, const char *endpoint) {
  void* sock = zmq_socket(ctx, ZMQ_SUB);
  zmq_connect(sock, endpoint);
  zmq_setsockopt(sock, ZMQ_SUBSCRIBE, "", 0);

  int timeout = 100;
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout, sizeof(int));

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
  auto s_sock = sub_sock(ctx, "ipc:///tmp/q1");
  auto p_sock = pub_sock(ctx, "ipc:///tmp/q0");
#else
  auto s_sock = sub_sock(ctx, "tcp://127.0.0.1:10004");
  auto p_sock = pub_sock(ctx, "tcp://*:10005");
#endif

  zmq_msg_t msg;
  zmq_msg_init_size (&msg, N);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < MSGS; i++){
    zmq_msg_send(&msg, p_sock, ZMQ_DONTWAIT);
    int r = zmq_msg_recv(&msg, s_sock, 0);
    if (r) {
      start = std::chrono::steady_clock::now();
      std::cout << "Timeout" << std::endl;
    }
  }
  auto end = std::chrono::steady_clock::now();


  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
  double throughput = ((double) MSGS / (double) elapsed);

  std::cout << "Elapsed: " << elapsed << " s" << std::endl;
  std::cout << "Throughput: " << throughput << " msg/s" << std::endl;

  zmq_close(p_sock);
  zmq_close(s_sock);
  return 0;
}
