#pragma once
#include "messaging.hpp"
class AlignedMessage {
 public:
  // if message is NULL,aligned_array.ptr is empty, msg_reader will return from constructor on the first line.
  AlignedMessage(Message *message) : aligned_array(message), msg_reader(aligned_array.ptr) { msg = message; }

  ~AlignedMessage() {
    if (msg) delete msg;
    if (aligned_array.buf) free(aligned_array.buf);
  }

  template <typename RootType>
  typename RootType::Reader getRoot() { return msg_reader.getRoot<RootType>(); }
  inline operator bool() { return msg != NULL; }
  inline const char *getData() { return msg->getData(); }
  inline size_t getSize() { return msg->getSize(); }

 private:
  struct _InnerAlignedArray {
    _InnerAlignedArray(Message *msg) {
      buf = NULL;
      if (msg == NULL) return;

      char *data = msg->getData();
      size_t size = msg->getSize();
      size_t buf_size = size;

      if (size % sizeof(capnp::word) != 0) buf_size = (size / sizeof(capnp::word) + 1) * sizeof(capnp::word);

      if (((reinterpret_cast<uintptr_t>(data)) % sizeof(capnp::word) != 0) || buf_size != size) {
        // malloc is defined as being word aligned
        buf = (char *)malloc(buf_size);
        memcpy(buf, data, size);
      }
      ptr = kj::ArrayPtr<const capnp::word>(reinterpret_cast<capnp::word *>(buf ? buf : data), buf_size / sizeof(capnp::word));
    }

    char *buf;
    kj::ArrayPtr<const capnp::word> ptr;
  };

  Message *msg;
  _InnerAlignedArray aligned_array;
  capnp::FlatArrayMessageReader msg_reader;
};
