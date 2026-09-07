#pragma once

#include "tools/cabana/streams/livestream.h"

#include <string>
#include <sys/types.h>

class DeviceStream : public LiveStream {
public:
  DeviceStream(std::string address = {});
  ~DeviceStream();
  // Run off the UI thread before handing a device stream to MainWindow.
  bool prepare(std::string &failure);
  void cancelPreparation() { exit_ = true; }
  inline std::string routeName() const override {
    return "Live Streaming From " + (zmq_address.empty() ? std::string("127.0.0.1") : zmq_address);
  }

protected:
  void start() override;
  void streamThread() override;
  void stopBridge();
  bool remote_ready_ = false;
  pid_t bridge_pid = -1;
  const std::string zmq_address;
};
