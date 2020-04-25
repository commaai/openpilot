#pragma once
#include <capnp/serialize.h>
#include <stdlib.h>

#include "cereal/gen/cpp/log.capnp.h"
#include "messaging.hpp"
#include "swaglog.h"
class AlignedMessage {
 public:
  AlignedMessage(Message *message) {
    buf = NULL;
    msg = message;
    if (!message) return;

    size_t size = message->getSize();
    size_t buf_size = size;

    if (size % sizeof(capnp::word) != 0) buf_size = (size / sizeof(capnp::word) + 1) * sizeof(capnp::word);

    if (((reinterpret_cast<uintptr_t>(message->getData())) % sizeof(capnp::word) != 0) || buf_size != size) {
      // malloc is defined as being word aligned
      buf = (char *)malloc(buf_size);
      memcpy(buf, message->getData(), size);
      LOGD("message is not aligned");
    }
    msg_reader = new capnp::FlatArrayMessageReader(kj::ArrayPtr<const capnp::word>(reinterpret_cast<capnp::word *>(buf ? buf : msg->getData()), buf_size / sizeof(capnp::word)));
    event = msg_reader->getRoot<cereal::Event>();
  }

  ~AlignedMessage() {
    if (msg) {
      delete msg;
      delete msg_reader;
    }
    if (buf) free(buf);
  }
  inline cereal::Event::Reader &getEvent() { return event; }
  inline operator bool() { return msg != NULL; }
  inline const char *getData() { return msg->getData(); }
  inline size_t getSize() { return msg->getSize(); }

 private:
  char *buf;
  Message *msg;
  capnp::FlatArrayMessageReader *msg_reader;
  cereal::Event::Reader event;
};
