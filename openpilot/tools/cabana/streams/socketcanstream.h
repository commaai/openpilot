#pragma once

#include <string>

#include "tools/cabana/streams/livestream.h"

struct SocketCanStreamConfig {
  std::string device = ""; // TODO: support multiple devices/buses at once
};

class SocketCanStream : public LiveStream {
public:
  SocketCanStream(SocketCanStreamConfig config_ = {});
  ~SocketCanStream();
  static bool available();

  inline std::string routeName() const override {
    return "Live Streaming From Socket CAN " + config.device;
  }

protected:
  void streamThread() override;
  bool connect();

  SocketCanStreamConfig config = {};
  int sock_fd = -1;
};
