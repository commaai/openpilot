#include "selfdrive/boardd/usbdevice.h"

#include <cassert>

#include "panda/board/panda.h"
#include "selfdrive/common/swaglog.h"

// USBContext

USBContext::USBContext() {
  int err = libusb_init(&context);
  if (err != 0) {
    LOGE("libusb initialization error %d", err);
    assert(0);
  }
#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(context, 3);
#endif
}

USBContext::~USBContext() {
  libusb_exit(context);
}

// USBDeviceList

USBDeviceList::USBDeviceList(const USBContext &ctx) {
  num_devices = libusb_get_device_list(ctx.context, &dev_list);
}

USBDeviceList::~USBDeviceList() {
  if (dev_list) libusb_free_device_list(dev_list, 1);
}

libusb_device_handle *USBDeviceList::open(const std::string &serial, std::string &out_serial) {
  for (ssize_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc = {};
    int ret = libusb_get_device_descriptor(dev_list[i], &desc);
    if (ret < 0 || desc.idVendor != USB_VID || desc.idProduct != USB_PID) continue;

    libusb_device_handle *handle = nullptr;
    if (libusb_open(dev_list[i], &handle) == 0) {
      char s[256] = {'\0'};
      libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, (uint8_t*)s, std::size(s) - 1);
      if (serial.empty() || serial == s) {
        out_serial = s;
        return handle;
      }
      libusb_close(handle);
    }
  }
  return nullptr;
}

int USBDeviceList::size() {
  int cnt = 0;
  for (ssize_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc = {};
    int ret = libusb_get_device_descriptor(dev_list[i], &desc);
    cnt += ret == 0 && desc.idVendor == USB_VID && desc.idProduct == USB_PID;
  }
  return cnt;
}
