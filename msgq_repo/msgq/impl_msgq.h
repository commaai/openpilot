#pragma once

#include <string>
#include <vector>

#include "msgq/ipc.h"
#include "msgq/msgq.h"

#define MAX_POLLERS 128

class MSGQSubSocket : public SubSocket {
private:
  msgq_queue_t * q = NULL;
  int timeout;
public:
  int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true, size_t segment_size=0);
  void setTimeout(int timeout);
  msgq_queue_t * getQueue() {return q;}
  Message *receive(bool non_blocking=false);
  ~MSGQSubSocket();
};

class MSGQPoller : public Poller {
private:
  std::vector<SubSocket*> sockets;
  msgq_pollitem_t polls[MAX_POLLERS];
  size_t num_polls = 0;

public:
  void registerSocket(SubSocket *socket);
  std::vector<SubSocket*> poll(int timeout);
  ~MSGQPoller(){}
};
