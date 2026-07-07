#pragma once

#include <sys/types.h>

#include <string>

#include "tools/cabana/streams/livestream.h"

class DeviceStream : public LiveStream {
public:
  DeviceStream(std::string address = {});
  ~DeviceStream();
  inline std::string routeName() const override {
    return "Live Streaming From " + (zmq_address.empty() ? std::string("127.0.0.1") : zmq_address);
  }

protected:
  void start() override;
  void streamThread() override;
  pid_t bridge_pid = -1;
  const std::string zmq_address;
};
