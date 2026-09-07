#pragma once

#include "tools/cabana/streams/livestream.h"

#include <string>

class DeviceStream : public LiveStream {
public:
  DeviceStream(std::string address = {});
  ~DeviceStream();
  inline std::string routeName() const override {
    return "Live Streaming From " + (zmq_address.empty() ? std::string("127.0.0.1") : zmq_address);
  }

protected:
  void streamThread() override;
  const std::string zmq_address;
};
