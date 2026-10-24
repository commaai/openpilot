#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <time.h>

#include "msgq/msgq.h"

#ifdef __APPLE__
#define CLOCK_BOOTTIME CLOCK_MONOTONIC
#endif

struct EventState;

class Message {
private:
  char * data = nullptr;
  size_t size = 0;
public:
  void init(size_t size);
  void init(char * data, size_t size);
  void takeOwnership(char * data, size_t size);
  void close();
  size_t getSize(){return size;}
  char * getData(){return data;}
  ~Message();
};

class SubSocket {
private:
  msgq_queue_t * q = nullptr;
  int timeout = -1;

  // fake mode (CEREAL_FAKE)
  EventState *fake_state = nullptr;
  int fake_fds[2] = {-1, -1};
  void open_fake_fifos();

public:
  int connect(std::string endpoint, bool conflate=false, size_t segment_size=0);
  void setTimeout(int timeout);
  msgq_queue_t * getQueue() {return q;}
  Message *receive(bool non_blocking=false);
  static SubSocket * create(std::string endpoint, bool conflate=false, size_t segment_size=0);
  ~SubSocket();
};

class PubSocket {
private:
  msgq_queue_t * q = nullptr;
public:
  int connect(std::string endpoint, size_t segment_size=0);
  int sendMessage(Message *message);
  int send(char *data, size_t size);
  bool all_readers_updated();
  static PubSocket * create(std::string endpoint, size_t segment_size=0);
  ~PubSocket();
};

class Poller {
private:
  bool fake;
  std::vector<SubSocket*> sockets;
  std::vector<msgq_pollitem_t> polls;

public:
  Poller();
  Poller(std::vector<SubSocket*> sockets);
  void registerSocket(SubSocket *socket);
  std::vector<SubSocket*> poll(int timeout);
};
