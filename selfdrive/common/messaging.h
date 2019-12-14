// the c version of cereal/messaging.py

#include <zmq.h>

// TODO: refactor to take in service instead of endpoint?
void *sub_sock(void *ctx, const char *endpoint) {
  void* sock = zmq_socket(ctx, ZMQ_SUB);
  assert(sock);
  zmq_setsockopt(sock, ZMQ_SUBSCRIBE, "", 0);
  int reconnect_ivl = 500;
  zmq_setsockopt(sock, ZMQ_RECONNECT_IVL_MAX, &reconnect_ivl, sizeof(reconnect_ivl));
  zmq_connect(sock, endpoint);
  return sock;
}

