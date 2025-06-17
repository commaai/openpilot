#ifndef __APPLE__
#include <sys/file.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "panda/board/comms_definitions.h"
#include "selfdrive/pandad/panda_comms.h"


#define SPI_SYNC 0x5AU
#define SPI_HACK 0x79U
#define SPI_DACK 0x85U
#define SPI_NACK 0x1FU
#define SPI_CHECKSUM_START 0xABU


enum SpiError {
  NACK = -2,
  ACK_TIMEOUT = -3,
};

const unsigned int SPI_ACK_TIMEOUT = 500; // milliseconds
const std::string SPI_DEVICE = "/dev/spidev0.0";

class LockEx {
public:
  LockEx(int fd, std::recursive_mutex &m) : fd(fd), m(m) {
    m.lock();
    flock(fd, LOCK_EX);
  }

  ~LockEx() {
    flock(fd, LOCK_UN);
    m.unlock();
  }

private:
  int fd;
  std::recursive_mutex &m;
};

#define SPILOG(fn, fmt, ...) do {  \
      fn(fmt, ## __VA_ARGS__);     \
      fn("  %d / 0x%x / %d / %d / tx: %s", \
         xfer_count, header.endpoint, header.tx_len, header.max_rx_len, \
         util::hexdump(tx_buf, std::min((int)header.tx_len, 8)).c_str()); \
      } while (0)

PandaSpiHandle::PandaSpiHandle(std::string serial) : PandaCommsHandle(serial) {
  int ret;
  const int uid_len = 12;
  uint8_t uid[uid_len] = {0};

  uint32_t spi_mode = SPI_MODE_0;
  uint8_t spi_bits_per_word = 8;

  // 50MHz is the max of the 845. note that some older
  // revs of the comma three may not support this speed
  uint32_t spi_speed = 50000000;

  if (!util::file_exists(SPI_DEVICE)) {
    goto fail;
  }

  spi_fd = open(SPI_DEVICE.c_str(), O_RDWR);
  if (spi_fd < 0) {
    LOGE("failed opening SPI device %d", spi_fd);
    goto fail;
  }

  // SPI settings
  ret = util::safe_ioctl(spi_fd, SPI_IOC_WR_MODE, &spi_mode);
  if (ret < 0) {
    LOGE("failed setting SPI mode %d", ret);
    goto fail;
  }

  ret = util::safe_ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed);
  if (ret < 0) {
    LOGE("failed setting SPI speed");
    goto fail;
  }

  ret = util::safe_ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &spi_bits_per_word);
  if (ret < 0) {
    LOGE("failed setting SPI bits per word");
    goto fail;
  }

  // get hw UID/serial
  ret = control_read(0xc3, 0, 0, uid, uid_len, 100);
  if (ret == uid_len) {
    std::stringstream stream;
    for (int i = 0; i < uid_len; i++) {
      stream << std::hex << std::setw(2) << std::setfill('0') << int(uid[i]);
    }
    hw_serial = stream.str();
  } else {
    LOGD("failed to get serial %d", ret);
    goto fail;
  }

  if (!serial.empty() && (serial != hw_serial)) {
    goto fail;
  }

  return;

fail:
  cleanup();
  throw std::runtime_error("Error connecting to panda");
}

PandaSpiHandle::~PandaSpiHandle() {
  std::lock_guard lk(hw_lock);
  cleanup();
}

void PandaSpiHandle::cleanup() {
  if (spi_fd != -1) {
    close(spi_fd);
    spi_fd = -1;
  }
}



int PandaSpiHandle::control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout) {
  ControlPacket_t packet = {
    .request = request,
    .param1 = param1,
    .param2 = param2,
    .length = 0
  };
  return spi_transfer_retry(0, (uint8_t *) &packet, sizeof(packet), NULL, 0, timeout);
}

int PandaSpiHandle::control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout) {
  ControlPacket_t packet = {
    .request = request,
    .param1 = param1,
    .param2 = param2,
    .length = length
  };
  return spi_transfer_retry(0, (uint8_t *) &packet, sizeof(packet), data, length, timeout);
}

int PandaSpiHandle::bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return bulk_transfer(endpoint, data, length, NULL, 0, timeout);
}
int PandaSpiHandle::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return bulk_transfer(endpoint, NULL, 0, data, length, timeout);
}

int PandaSpiHandle::bulk_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t rx_len, unsigned int timeout) {
  const int xfer_size = SPI_BUF_SIZE - 0x40;

  int ret = 0;
  uint16_t length = (tx_data != NULL) ? tx_len : rx_len;
  for (int i = 0; i < (int)std::ceil((float)length / xfer_size); i++) {
    int d;
    if (tx_data != NULL) {
      int len = std::min(xfer_size, tx_len - (xfer_size * i));
      d = spi_transfer_retry(endpoint, tx_data + (xfer_size * i), len, NULL, 0, timeout);
    } else {
      uint16_t to_read = std::min(xfer_size, rx_len - ret);
      d = spi_transfer_retry(endpoint, NULL, 0, rx_data + (xfer_size * i), to_read, timeout);
    }

    if (d < 0) {
      SPILOG(LOGE, "SPI: bulk transfer failed with %d", d);
      comms_healthy = false;
      return d;
    }

    ret += d;
    if ((rx_data != NULL) && d < xfer_size) {
      break;
    }
  }

  return ret;
}

std::vector<std::string> PandaSpiHandle::list() {
  try {
    PandaSpiHandle sh("");
    return {sh.hw_serial};
  } catch (std::exception &e) {
    // no panda on SPI
  }
  return {};
}

void add_checksum(uint8_t *data, int data_len) {
  data[data_len] = SPI_CHECKSUM_START;
  for (int i=0; i < data_len; i++) {
    data[data_len] ^= data[i];
  }
}

bool check_checksum(uint8_t *data, int data_len) {
  uint8_t checksum = SPI_CHECKSUM_START;
  for (uint16_t i = 0U; i < data_len; i++) {
    checksum ^= data[i];
  }
  return checksum == 0U;
}


int PandaSpiHandle::spi_transfer_retry(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout) {
  int ret;
  int nack_count = 0;
  int timeout_count = 0;
  bool timed_out = false;
  double start_time = millis_since_boot();

  do {
    ret = spi_transfer(endpoint, tx_data, tx_len, rx_data, max_rx_len, timeout);

    if (ret < 0) {
      timed_out = (timeout != 0) && (timeout_count > 5);
      timeout_count += ret == SpiError::ACK_TIMEOUT;

      // give other threads a chance to run
      std::this_thread::yield();

      if (ret == SpiError::NACK) {
        // prevent busy waiting while the panda is NACK'ing
        // due to full TX buffers
        nack_count += 1;
        if (nack_count > 3) {
          SPILOG(LOGD, "NACK sleep %d", nack_count);
          usleep(std::clamp(nack_count*10, 200, 2000));
        }
      }
    }
  } while (ret < 0 && connected && !timed_out);

  if (ret < 0) {
    SPILOG(LOGE, "transfer failed, after %d tries, %.2fms", timeout_count, millis_since_boot() - start_time);
  }

  return ret;
}

int PandaSpiHandle::wait_for_ack(uint8_t ack, uint8_t tx, unsigned int timeout, unsigned int length) {
  double start_millis = millis_since_boot();
  if (timeout == 0) {
    timeout = SPI_ACK_TIMEOUT;
  }
  timeout = std::clamp(timeout, 20U, SPI_ACK_TIMEOUT);

  spi_ioc_transfer transfer = {
    .tx_buf = (uint64_t)tx_buf,
    .rx_buf = (uint64_t)rx_buf,
    .len = length,
  };
  memset(tx_buf, tx, length);

  while (true) {
    int ret = lltransfer(transfer);
    if (ret < 0) {
      SPILOG(LOGE, "SPI: failed to send ACK request");
      return ret;
    }

    if (rx_buf[0] == ack) {
      break;
    } else if (rx_buf[0] == SPI_NACK) {
      SPILOG(LOGD, "SPI: got NACK, waiting for 0x%x", ack);
      return SpiError::NACK;
    }

    // handle timeout
    if (millis_since_boot() - start_millis > timeout) {
      SPILOG(LOGW, "SPI: timed out waiting for ACK, waiting for 0x%x", ack);
      return SpiError::ACK_TIMEOUT;
    }
  }

  return 0;
}

int PandaSpiHandle::lltransfer(spi_ioc_transfer &t) {
  static const double err_prob = std::stod(util::getenv("SPI_ERR_PROB", "-1"));

  if (err_prob > 0) {
    if ((static_cast<double>(rand()) / RAND_MAX) < err_prob) {
      printf("transfer len error\n");
      t.len = rand() % SPI_BUF_SIZE;
    }
    if ((static_cast<double>(rand()) / RAND_MAX) < err_prob && t.tx_buf != (uint64_t)NULL) {
      printf("corrupting TX\n");
      for (int i = 0; i < t.len; i++) {
        if ((static_cast<double>(rand()) / RAND_MAX) > 0.9) {
          ((uint8_t*)t.tx_buf)[i] = (uint8_t)(rand() % 256);
        }
      }
    }
  }

  int ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t);

  if (err_prob > 0) {
    if ((static_cast<double>(rand()) / RAND_MAX) < err_prob && t.rx_buf != (uint64_t)NULL) {
      printf("corrupting RX\n");
      for (int i = 0; i < t.len; i++) {
        if ((static_cast<double>(rand()) / RAND_MAX) > 0.9) {
          ((uint8_t*)t.rx_buf)[i] = (uint8_t)(rand() % 256);
        }
      }
    }
  }

  return ret;
}

int PandaSpiHandle::spi_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout) {
  int ret;
  uint16_t rx_data_len;
  LockEx lock(spi_fd, hw_lock);

  // needs to be less, since we need to have space for the checksum
  assert(tx_len < SPI_BUF_SIZE);
  assert(max_rx_len < SPI_BUF_SIZE);

  xfer_count++;
  header = {
    .sync = SPI_SYNC,
    .endpoint = endpoint,
    .tx_len = tx_len,
    .max_rx_len = max_rx_len
  };

  spi_ioc_transfer transfer = {
    .tx_buf = (uint64_t)tx_buf,
    .rx_buf = (uint64_t)rx_buf
  };

  // Send header
  memcpy(tx_buf, &header, sizeof(header));
  add_checksum(tx_buf, sizeof(header));
  transfer.len = sizeof(header) + 1;
  ret = lltransfer(transfer);
  if (ret < 0) {
    SPILOG(LOGE, "SPI: failed to send header");
    goto fail;
  }

  // Wait for (N)ACK
  ret = wait_for_ack(SPI_HACK, 0x11, timeout, 1);
  if (ret < 0) {
    goto fail;
  }

  // Send data
  if (tx_data != NULL) {
    memcpy(tx_buf, tx_data, tx_len);
  }
  add_checksum(tx_buf, tx_len);
  transfer.len = tx_len + 1;
  ret = lltransfer(transfer);
  if (ret < 0) {
    SPILOG(LOGE, "SPI: failed to send data");
    goto fail;
  }

  // Wait for (N)ACK
  ret = wait_for_ack(SPI_DACK, 0x13, timeout, 3);
  if (ret < 0) {
    goto fail;
  }

  // Read data
  rx_data_len = *(uint16_t *)(rx_buf+1);
  if (rx_data_len >= SPI_BUF_SIZE) {
    SPILOG(LOGE, "SPI: RX data len larger than buf size %d", rx_data_len);
    goto fail;
  }

  transfer.len = rx_data_len + 1;
  transfer.rx_buf = (uint64_t)(rx_buf + 2 + 1);
  ret = lltransfer(transfer);
  if (ret < 0) {
    SPILOG(LOGE, "SPI: failed to read rx data");
    goto fail;
  }
  if (!check_checksum(rx_buf, rx_data_len + 4)) {
    SPILOG(LOGE, "SPI: bad checksum");
    goto fail;
  }

  if (rx_data != NULL) {
    memcpy(rx_data, rx_buf + 3, rx_data_len);
  }

  return rx_data_len;

fail:
  // ensure slave is in a consistent state
  // and ready for the next transfer
  int nack_cnt = 0;
  while (nack_cnt < 3) {
    if (wait_for_ack(SPI_NACK, 0x14, 1, SPI_BUF_SIZE/2) == 0) {
      nack_cnt += 1;
    } else {
      nack_cnt = 0;
    }
  }

  if (ret >= 0) ret = -1;
  return ret;
}
#endif
