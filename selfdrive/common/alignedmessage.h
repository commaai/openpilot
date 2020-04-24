#pragma once
#include <stdlib.h>
#include <capnp/serialize.h>
#include "messaging.hpp"
#include "swaglog.h"
class AlignedMessage{
  protected:
    char *  buf;
    size_t  buf_size;
    bool    allocated;
    Message *msg;

  public:
    AlignedMessage(Message* message){
      allocated = false;
      msg = message;
      if (!message) return;

      char *data = buf = message->getData();
      size_t size = buf_size = message->getSize();
            
      if (size % sizeof(capnp::word) !=0) buf_size = (size / sizeof(capnp::word) + 1) * sizeof(capnp::word);
            
      if (((reinterpret_cast<uintptr_t>(data)) % sizeof(capnp::word) != 0) || buf_size != size){
        // malloc is defined as being word aligned
        buf = (char *)malloc(buf_size);
        memcpy(buf, data, size);
        allocated = true;
        LOGD("message is not aligned");
      }
    }

    ~AlignedMessage(){
      if (msg) delete msg;
      if (allocated) free(buf);
    }
    inline operator kj::ArrayPtr<const capnp::word>(){return kj::ArrayPtr<const capnp::word>(reinterpret_cast<capnp::word*>(buf), buf_size / sizeof(capnp::word));}
    inline operator bool(){return msg != NULL;}
    inline const char* getData(){return buf;}
    inline size_t getSize(){return msg->getSize();}
};
