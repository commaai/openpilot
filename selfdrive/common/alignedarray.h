#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <capnp/serialize.h>
class AlignedArray{
  protected:
    char *  buf;
    size_t  buf_size;
    bool    allocated;

  public:
    AlignedArray(char* data, size_t size){
      buf = data;
      buf_size = size;
      if (size % sizeof(capnp::word) !=0){
        buf_size = (size / sizeof(capnp::word) + 1) * sizeof(capnp::word);
      }
      allocated = false;
      
      if (((reinterpret_cast<uintptr_t>(data)) % sizeof(capnp::word) != 0) || buf_size != size){
        // macOS, OpenBSD, and Android have posix_memalign().
        int error = posix_memalign(reinterpret_cast<void **>(&buf), sizeof(capnp::word), buf_size);
        assert(error == 0);
        memcpy(buf, data, size);
        allocated = true;
        printf("data is not aligned\n");
      }
    }

    inline operator kj::ArrayPtr<const capnp::word>(){
      return kj::ArrayPtr<const capnp::word>(reinterpret_cast<capnp::word*>(buf), buf_size / sizeof(capnp::word));
    }

    ~AlignedArray(){
      if (allocated){
        free(buf);
      }
    }
};
