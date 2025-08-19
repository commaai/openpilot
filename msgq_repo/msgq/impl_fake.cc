#include "msgq/impl_fake.h"

void FakePoller::registerSocket(SubSocket *socket) {
  this->sockets.push_back(socket);
}

std::vector<SubSocket*> FakePoller::poll(int timeout) {
  return this->sockets;
}
