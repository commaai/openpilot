#pragma once

#include <libusb-1.0/libusb.h>

#include <string>

struct USBContext {
  USBContext();
  ~USBContext();
  libusb_context *context;
};

class USBDeviceList {
public:
  USBDeviceList(const USBContext &ctx);
  ~USBDeviceList();
  libusb_device_handle *open(const std::string &serial, std::string &out_serial);
  int size();

private:
  libusb_device **dev_list = nullptr;
  ssize_t num_devices = 0;
};
