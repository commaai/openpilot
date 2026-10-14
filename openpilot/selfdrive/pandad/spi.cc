#include <sys/file.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "panda/board/comms_definitions.h"
#include "selfdrive/pandad/panda_comms.h"


// protocol, see panda/board/drivers/spi.h
#define SPI_SLAVE_READY 0x3CU
#define SPI_ACK 0xC3U
#define SPI_NACK 0x5AU
#define SPI_POLL 0xFFU  // filler while reading the response, never a valid request

#define SPI_CRC_SIZE 4U
#define SPI_XFER_SIZE (SPI_BUF_SIZE - 0x40)  // bulk chunk size, leaves room for the protocol overhead
#define SPI_FIRST_READ_SIZE (1024 - sizeof(spi_resp_header) - SPI_CRC_SIZE)  // reads are usually small, start small


enum SpiError {
  NACK = -2,
  ACK_TIMEOUT = -3,
  SILENT = -4,
};

const unsigned int SPI_MIN_TIMEOUT = 100; // milliseconds
const double SPI_SILENT_TIMEOUT = 1000;  // milliseconds, longer than the slowest request takes on the panda
const std::string SPI_DEVICE = "/dev/spidev0.0";

class LockEx {
public:
  LockEx(int fd_, std::recursive_mutex &m_) : fd(fd_), m(m_) {
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
         xfer_count, header.endpoint, header.tx_len, header.rx_len, \
         util::hexdump(tx_buf + sizeof(header), std::min((int)header.tx_len, 8)).c_str()); \
      } while (0)

PandaSpiHandle::PandaSpiHandle(std::string serial) {
  int ret;
  const int uid_len = 12;
  uint8_t uid[uid_len] = {0};

  uint32_t spi_mode = SPI_MODE_0;
  uint8_t spi_bits_per_word = 8;

  // 50MHz is the max of the 845. note that some older
  // revs of the comma three may not support this speed
  uint32_t spi_speed = 50000000;
  try {
    if (!util::file_exists(SPI_DEVICE)) {
      throw std::runtime_error("Error connecting to panda: SPI device not found");
    }

    spi_fd = open(SPI_DEVICE.c_str(), O_RDWR);
    if (spi_fd < 0) {
      LOGE("failed opening SPI device %d", spi_fd);
      throw std::runtime_error("Error connecting to panda: failed to open SPI device");
    }

    // SPI settings
    util::safe_ioctl(spi_fd, SPI_IOC_WR_MODE, &spi_mode, "failed setting SPI mode");
    util::safe_ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed, "failed setting SPI speed");
    util::safe_ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &spi_bits_per_word, "failed setting SPI bits per word");

    // get hw UID/serial
    ret = control_read(0xc3, 0, 0, uid, uid_len, 500);
    if (ret == uid_len) {
      std::stringstream stream;
      for (int i = 0; i < uid_len; i++) {
        stream << std::hex << std::setw(2) << std::setfill('0') << int(uid[i]);
      }
      hw_serial = stream.str();
    } else {
      LOGD("failed to get serial %d", ret);
      throw std::runtime_error("Error connecting to panda: failed to get serial");
    }

    if (!serial.empty() && (serial != hw_serial)) {
      throw std::runtime_error("Error connecting to panda: serial mismatch");
    }

  } catch (...) {
    cleanup();
    throw;
  }
  return;
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
  for (int pos = 0; pos < length; pos += SPI_XFER_SIZE) {
    int ret = spi_transfer_retry(endpoint, data + pos, std::min(SPI_XFER_SIZE, length - pos), NULL, 0, timeout);
    if (ret < 0) {
      comms_healthy = false;
      return ret;
    }
  }
  return length;
}

int PandaSpiHandle::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int pos = 0;
  while (pos < length) {
    int to_read = std::min(pos == 0 ? (int)SPI_FIRST_READ_SIZE : SPI_XFER_SIZE, length - pos);
    int ret = spi_transfer_retry(endpoint, NULL, 0, data + pos, to_read, timeout);
    if (ret < 0) {
      comms_healthy = false;
      return ret;
    }
    pos += ret;
    if (ret < to_read) {
      break;
    }
  }
  return pos;
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

// random, so it's never the same as another request the panda just handled, even from another client
static uint32_t random_id() {
  static thread_local std::mt19937 rng{std::random_device{}()};
  return rng();
}

// standard CRC-32, same as zlib's crc32()
static uint32_t crc32(const uint8_t *data, uint32_t len) {
  static const auto table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int j = 0; j < 8; j++) {
        c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
      }
      t[i] = c;
    }
    return t;
  }();

  uint32_t crc = 0xFFFFFFFFU;
  for (uint32_t i = 0; i < len; i++) {
    crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFU;
}

static void put_crc(uint8_t *data, uint32_t len) {
  uint32_t crc = crc32(data, len);
  memcpy(&data[len], &crc, sizeof(crc));
}

static bool check_crc(const uint8_t *data, uint32_t len) {
  uint32_t crc;
  memcpy(&crc, &data[len], sizeof(crc));
  return crc == crc32(data, len);
}


int PandaSpiHandle::spi_transfer_retry(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout) {
  // timeout of 0 means forever
  const double deadline = (timeout == 0) ? INFINITY : millis_since_boot() + std::max(timeout, SPI_MIN_TIMEOUT);

  // the id stays the same when we send the request again, so the panda never runs it twice
  uint32_t id = random_id();
  int ret;
  int nack_count = 0;
  while (true) {
    ret = spi_transfer(endpoint, id, tx_data, tx_len, rx_data, max_rx_len, deadline);
    if (ret >= 0 || !connected) {
      break;
    }

    if (ret == SpiError::NACK) {
      // the panda NACKs while its CAN TX buffers are full. that's not a comms
      // failure, so keep trying regardless of the timeout, without busy waiting.
      // a new id, since the panda would answer the same id with the same NACK
      id = random_id();
      nack_count += 1;
      if (nack_count > 3) {
        usleep(std::clamp(nack_count*10, 200, 2000));
      }
    } else if (ret == SpiError::SILENT) {
      // the panda went quiet. let go of the bus for a bit, someone else might need it
      LOGW("SPI: panda isn't responding");
      usleep(10000);
    } else if (millis_since_boot() > deadline) {
      break;
    }

    // give other threads a chance to run
    std::this_thread::yield();
  }

  if (ret < 0) {
    SPILOG(LOGE, "SPI: transfer failed with %d", ret);
  }
  return ret;
}

int PandaSpiHandle::lltransfer(spi_ioc_transfer &t) {
  static const double err_prob = std::stod(util::getenv("SPI_ERR_PROB", "-1"));
  auto chance = [](double p) { return (static_cast<double>(rand()) / RAND_MAX) < p; };
  auto corrupt = [&](uint64_t buf) {
    for (uint32_t i = 0; i < t.len; i++) {
      if (chance(0.1)) ((uint8_t*)buf)[i] = (uint8_t)(rand() % 256);
    }
  };

  if (err_prob > 0 && chance(err_prob)) {
    printf("transfer len error\n");
    t.len = rand() % SPI_BUF_SIZE;
  }
  if (err_prob > 0 && chance(err_prob) && t.tx_buf == (uint64_t)tx_buf) {
    printf("corrupting TX\n");
    corrupt(t.tx_buf);
  }

  int ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t);
  if (err_prob > 0 && chance(err_prob)) {
    printf("ioctl error\n");
    ret = -1;
  }

  if (err_prob > 0 && chance(err_prob)) {
    printf("corrupting RX\n");
    corrupt(t.rx_buf);
  }
  return ret;
}

// One request and its response. Resends until the panda got the request and we got its response,
// or the deadline. Every transfer either sees all of what the panda queued from its first byte, or
// nothing it can mistake for a response. Returns the response length, or a negative SpiError
int PandaSpiHandle::spi_transfer(uint8_t endpoint, uint32_t id, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, double deadline) {
  static const auto poll = [] {
    std::array<uint8_t, SPI_BUF_SIZE> p;
    p.fill(SPI_POLL);
    return p;
  }();

  LockEx lock(spi_fd, hw_lock);

  // the response is always read in one transfer of rx_len
  const uint32_t rx_len = sizeof(spi_resp_header) + max_rx_len + SPI_CRC_SIZE;
  const uint32_t req_len = sizeof(header) + tx_len;
  assert(req_len + SPI_CRC_SIZE <= SPI_BUF_SIZE);
  assert(rx_len <= SPI_BUF_SIZE);

  xfer_count++;
  header = {
    .endpoint = endpoint,
    .id = id,
    .tx_len = tx_len,
    .rx_len = (uint16_t)rx_len,
  };

  auto xfer = [&](const uint8_t *tx, uint32_t len) {
    spi_ioc_transfer t = {.tx_buf = (uint64_t)tx, .rx_buf = (uint64_t)rx_buf, .len = len};
    int ret = lltransfer(t);
    if (ret < 0) {
      SPILOG(LOGE, "SPI: transfer failed");
    }
    return ret;
  };

  // when the panda last showed it's there. while it's quiet, we let go of the bus
  double alive = millis_since_boot();
  while (true) {
    // send the request, until the panda was ready for it. it's rebuilt every time,
    // since SPI_ERR_PROB corrupts tx_buf
    memcpy(tx_buf, &header, sizeof(header));
    if (tx_data != NULL) {
      memcpy(tx_buf + sizeof(header), tx_data, tx_len);
    }
    put_crc(tx_buf, req_len);
    // a failed transfer is sent again like any other, keeping the bus, so nothing can run in between
    if (xfer(tx_buf, req_len + SPI_CRC_SIZE) >= 0 && rx_buf[0] == SPI_SLAVE_READY) {
      alive = millis_since_boot();
      // read until the response. if the panda is ready again, it never got the request
      do {
        if (xfer(poll.data(), rx_len) < 0) {
          break;
        }
        if (rx_buf[0] == SPI_SLAVE_READY) {
          alive = millis_since_boot();
          break;
        }

        // a response that got cut off or corrupted isn't one, keep reading
        spi_resp_header resp;
        memcpy(&resp, rx_buf, sizeof(resp));
        if ((resp.status == SPI_ACK || resp.status == SPI_NACK) && resp.len <= max_rx_len && check_crc(rx_buf, sizeof(resp) + resp.len)) {
          // otherwise it's the response to an old request, and we send ours again
          if (resp.id == header.id) {
            if (resp.status == SPI_NACK) {
              SPILOG(LOGD, "SPI: got NACK");
              return SpiError::NACK;
            }
            if (rx_data != NULL) {
              memcpy(rx_data, rx_buf + sizeof(resp), resp.len);
            }
            return resp.len;
          }
          alive = millis_since_boot();
          break;
        }
      } while (millis_since_boot() < std::min(deadline, alive + SPI_SILENT_TIMEOUT));
    }

    if (millis_since_boot() > deadline) {
      SPILOG(LOGW, "SPI: timed out");
      return SpiError::ACK_TIMEOUT;
    }
    if (millis_since_boot() > alive + SPI_SILENT_TIMEOUT) {
      return SpiError::SILENT;
    }
  }
}
