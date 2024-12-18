#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cerrno>

#include "msgq/impl_msgq.h"


volatile sig_atomic_t msgq_do_exit = 0;

void sig_handler(int signal) {
  assert(signal == SIGINT || signal == SIGTERM);
  msgq_do_exit = 1;
}


MSGQContext::MSGQContext() {
}

MSGQContext::~MSGQContext() {
}

void MSGQMessage::init(size_t sz) {
  size = sz;
  data = new char[size];
}

void MSGQMessage::init(char * d, size_t sz) {
  size = sz;
  data = new char[size];
  memcpy(data, d, size);
}

void MSGQMessage::takeOwnership(char * d, size_t sz) {
  size = sz;
  data = d;
}

void MSGQMessage::close() {
  if (size > 0){
    delete[] data;
  }
  size = 0;
}

MSGQMessage::~MSGQMessage() {
  this->close();
}

int MSGQSubSocket::connect(Context *context, std::string endpoint, std::string address, bool conflate, bool check_endpoint){
  assert(context);
  assert(address == "127.0.0.1");

  q = new msgq_queue_t;
  int r = msgq_new_queue(q, endpoint.c_str(), DEFAULT_SEGMENT_SIZE);
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
  msgq_do_exit = 0;

  void (*prev_handler_sigint)(int);
  void (*prev_handler_sigterm)(int);
  if (!non_blocking){
    prev_handler_sigint = std::signal(SIGINT, sig_handler);
    prev_handler_sigterm = std::signal(SIGTERM, sig_handler);
  }

  msgq_msg_t msg;

  MSGQMessage *r = NULL;

  int rc = msgq_msg_recv(&msg, q);

  // Hack to implement blocking read with a poller. Don't use this
  while (!non_blocking && rc == 0 && msgq_do_exit == 0){
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


  if (!non_blocking){
    std::signal(SIGINT, prev_handler_sigint);
    std::signal(SIGTERM, prev_handler_sigterm);
  }

  errno = msgq_do_exit ? EINTR : 0;

  if (rc > 0){
    if (msgq_do_exit){
      msgq_msg_close(&msg); // Free unused message on exit
    } else {
      r = new MSGQMessage;
      r->takeOwnership(msg.data, msg.size);
    }
  }

  return (Message*)r;
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

int MSGQPubSocket::connect(Context *context, std::string endpoint, bool check_endpoint){
  assert(context);

  // TODO
  //if (check_endpoint && !service_exists(std::string(endpoint))){
  //  std::cout << "Warning, " << std::string(endpoint) << " is not in service list." << std::endl;
  //}

  q = new msgq_queue_t;
  int r = msgq_new_queue(q, endpoint.c_str(), DEFAULT_SEGMENT_SIZE);
  if (r != 0){
    return r;
  }

  msgq_init_publisher(q);

  return 0;
}

int MSGQPubSocket::sendMessage(Message *message){
  msgq_msg_t msg;
  msg.data = message->getData();
  msg.size = message->getSize();

  return msgq_msg_send(&msg, q);
}

int MSGQPubSocket::send(char *data, size_t size){
  msgq_msg_t msg;
  msg.data = data;
  msg.size = size;

  return msgq_msg_send(&msg, q);
}

bool MSGQPubSocket::all_readers_updated() {
  return msgq_all_readers_updated(q);
}

MSGQPubSocket::~MSGQPubSocket(){
  if (q != NULL){
    msgq_close_queue(q);
    delete q;
  }
}


void MSGQPoller::registerSocket(SubSocket * socket){
  assert(num_polls + 1 < MAX_POLLERS);
  polls[num_polls].q = (msgq_queue_t*)socket->getRawSocket();

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
