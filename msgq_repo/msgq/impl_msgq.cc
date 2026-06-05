#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <string>
#include <vector>

#include "msgq/impl_msgq.h"

void Message::init(size_t sz) {
  size = sz;
  data = new char[size];
}

void Message::init(char * d, size_t sz) {
  size = sz;
  data = new char[size];
  memcpy(data, d, size);
}

void Message::takeOwnership(char * d, size_t sz) {
  size = sz;
  data = d;
}

void Message::close() {
  if (size > 0){
    delete[] data;
  }
  size = 0;
}

Message::~Message() {
  this->close();
}

int MSGQSubSocket::connect(Context *context, std::string endpoint, std::string address, bool conflate, bool check_endpoint, size_t segment_size){
  assert(context);
  assert(address == "127.0.0.1");

  q = new msgq_queue_t;
  size_t size = segment_size > 0 ? segment_size : DEFAULT_SEGMENT_SIZE;
  int r = msgq_new_queue(q, endpoint.c_str(), size);
  if (r != 0){
    return r;
  }

  msgq_init_subscriber(q);

  if (conflate){
    q->read_conflate = true;
  }

  timeout = -1;

  return 0;
}


Message * MSGQSubSocket::receive(bool non_blocking){
  msgq_msg_t msg;

  Message *r = NULL;

  int rc = msgq_msg_recv(&msg, q);

  // Hack to implement blocking read with a poller. Don't use this
  while (!non_blocking && rc == 0){
    msgq_pollitem_t items[1];
    items[0].q = q;

    int t = (timeout != -1) ? timeout : 100;

    int n = msgq_poll(items, 1, t);
    rc = msgq_msg_recv(&msg, q);

    // The poll indicated a message was ready, but the receive failed. Try again
    if (n == 1 && rc == 0){
      continue;
    }

    if (timeout != -1){
      break;
    }
  }

  if (rc > 0){
    r = new Message;
    r->takeOwnership(msg.data, msg.size);
  }

  return r;
}

void MSGQSubSocket::setTimeout(int t){
  timeout = t;
}

MSGQSubSocket::~MSGQSubSocket(){
  if (q != NULL){
    msgq_close_queue(q);
    delete q;
  }
}

int PubSocket::connect(Context *context, std::string endpoint, bool check_endpoint, size_t segment_size){
  assert(context);

  q = new msgq_queue_t;
  size_t size = segment_size > 0 ? segment_size : DEFAULT_SEGMENT_SIZE;
  int r = msgq_new_queue(q, endpoint.c_str(), size);
  if (r != 0){
    return r;
  }

  msgq_init_publisher(q);

  return 0;
}

int PubSocket::sendMessage(Message *message){
  msgq_msg_t msg;
  msg.data = message->getData();
  msg.size = message->getSize();

  return msgq_msg_send(&msg, q);
}

int PubSocket::send(char *data, size_t size){
  msgq_msg_t msg;
  msg.data = data;
  msg.size = size;

  return msgq_msg_send(&msg, q);
}

bool PubSocket::all_readers_updated() {
  return msgq_all_readers_updated(q);
}

PubSocket::~PubSocket(){
  if (q != NULL){
    msgq_close_queue(q);
    delete q;
  }
}


void MSGQPoller::registerSocket(SubSocket * socket){
  assert(num_polls + 1 < MAX_POLLERS);
  polls[num_polls].q = static_cast<MSGQSubSocket*>(socket)->getQueue();

  sockets.push_back(socket);
  num_polls++;
}

std::vector<SubSocket*> MSGQPoller::poll(int timeout){
  std::vector<SubSocket*> r;

  msgq_poll(polls, num_polls, timeout);
  for (size_t i = 0; i < num_polls; i++){
    if (polls[i].revents){
      r.push_back(sockets[i]);
    }
  }

  return r;
}
