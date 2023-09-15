#pragma once

#include <string>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "cereal/messaging/msgq.h"

#define MAX_POLLERS 128

class MSGQContext : public Context {
private:
  void * context = NULL;
public:
  MSGQContext();
  void * getRawContext() {return context;}
  ~MSGQContext();
};

class MSGQMessage : public Message {
private:
  char * data;
  size_t size;
public:
  void init(size_t size);
  void init(char *data, size_t size);
  void takeOwnership(char *data, size_t size);
  size_t getSize(){return size;}
  char * getData(){return data;}
  void close();
  ~MSGQMessage();
};

class MSGQSubSocket : public SubSocket {
private:
  msgq_queue_t * q = NULL;
  int timeout;
public:
  int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true);
  void setTimeout(int timeout);
  void * getRawSocket() {return (void*)q;}
  Message *receive(bool non_blocking=false);
  ~MSGQSubSocket();
};

class MSGQPubSocket : public PubSocket {
private:
  msgq_queue_t * q = NULL;
public:
  int connect(Context *context, std::string endpoint, bool check_endpoint=true);
  int sendMessage(Message *message);
  int send(char *data, size_t size);
  bool all_readers_updated();
  ~MSGQPubSocket();
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
