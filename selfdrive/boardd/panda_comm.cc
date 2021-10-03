#include "selfdrive/boardd/panda_comm.h"

#include <cassert>
#include "selfdrive/common/swaglog.h"

namespace {

struct UsbContext {
  UsbContext() {
    int err = libusb_init(&ctx);
    assert(err == 0);

#if LIBUSB_API_VERSION >= 0x01000106
    libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
    libusb_set_debug(ctx, 3);
#endif
  }

  ~UsbContext() { libusb_exit(ctx); }

  libusb_context *ctx = nullptr;
};

libusb_context *libUsbContext() {
  static UsbContext context;
  return context.ctx;
}

class UsbDeviceList {
public:
  UsbDeviceList() {
    num_devices = libusb_get_device_list(libUsbContext(), &dev_list);
    if (num_devices < 0) {
      LOGE("libusb can't get device list");
    }
  }

  ~UsbDeviceList() {
    if (dev_list) libusb_free_device_list(dev_list, 1);
  }

  auto list(uint16_t vid, uint16_t pid) {
    std::vector<std::pair<libusb_device *, std::string>> result;

    for (size_t i = 0; i < num_devices; ++i) {
      libusb_device_descriptor desc = {};
      libusb_device_handle *handle = nullptr;
      int ret = libusb_get_device_descriptor(dev_list[i], &desc);
      if (ret == 0 && desc.idVendor == vid && desc.idProduct == pid && libusb_open(dev_list[i], &handle) == 0) {
        unsigned char serial[256] = {'\0'};
        libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, serial, std::size(serial) - 1);
        result.push_back({dev_list[i], (const char *)serial});
        libusb_close(handle);
      }
    }

    return result;
  }

private:
  libusb_device **dev_list = nullptr;
  ssize_t num_devices = 0;
};

}  // namespace


PandaComm::PandaComm(uint16_t vid, uint16_t pid, const std::string &serial) {
  UsbDeviceList device_list;
  for (auto &[dev, device_serial] : device_list.list(vid, pid)) {
    if (serial.empty() || serial == device_serial) {
      libusb_open(dev, &dev_handle);
      break;
    }
  }
  if (dev_handle) {
    if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
      libusb_detach_kernel_driver(dev_handle, 0);
    }
    if (libusb_set_configuration(dev_handle, 1) != 0 || libusb_claim_interface(dev_handle, 0) != 0) {
      libusb_close(dev_handle);
      dev_handle = nullptr;
    }
  }
  if (!dev_handle) {
    throw std::runtime_error("Error connecting to panda");
  }
}

PandaComm::~PandaComm() {
  libusb_release_interface(dev_handle, 0);
  libusb_close(dev_handle);
}

std::vector<std::string> PandaComm::list() {
  std::vector<std::string> serials;
  UsbDeviceList device_list;
  for (auto &[dev, serial] : device_list.list(PANDA_VENDOR_ID, PANDA_PRODUCT_ID)) {
    serials.push_back(serial);
  }
  return serials;
}

int PandaComm::usb_transfer(libusb_endpoint_direction dir, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  const uint8_t bmRequestType = dir | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;
  int r = LIBUSB_ERROR_NO_DEVICE;
  std::lock_guard lk(usb_lock);
  while (connected) {
    r = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (r >= 0) break;

    handle_usb_issue(r, __func__);
  }
  return r;
}

int PandaComm::usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  return usb_transfer(LIBUSB_ENDPOINT_OUT, bRequest, wValue, wIndex, timeout);
}

int PandaComm::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  return usb_transfer(LIBUSB_ENDPOINT_IN, bRequest, wValue, wIndex, timeout);
}

int PandaComm::usb_bulk_write(unsigned char endpoint, unsigned char *data, int length, unsigned int timeout) {
  int transferred = 0;
  std::lock_guard lk(usb_lock);
  while (connected) {
    // Try sending can messages. If the receive buffer on the panda is full it will NAK
    // and libusb will try again. After 5ms, it will time out. We will drop the messages.
    int err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    } else if (err != 0 || length != transferred) {
      handle_usb_issue(err, __func__);
    }
  };
  return transferred;
}

int PandaComm::usb_bulk_read(unsigned char endpoint, unsigned char *data, int length, unsigned int timeout) {
  int transferred = 0;
  std::lock_guard lk(usb_lock);
  while (connected) {
    int err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      break;  // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }
  };
  return transferred;
}

void PandaComm::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}
