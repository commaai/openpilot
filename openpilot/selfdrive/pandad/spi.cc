#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <array>
#include <cerrno>
#include <limits>

#ifdef SPI_STRESS_HOOKS
#include "stress_hooks.h"
#endif
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
#include "panda/board/spi_protocol.h"
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
// TODO: fix SPI turnaround synchronization at the protocol level.
static uint64_t spi_last_bus_activity_ns = 0;  // protected by hw_lock

static void wait_for_spi_turnaround(uint64_t start_ns) {
  while ((nanos_since_boot() - start_ns) < 400000) {}
}

class LockEx {
public:
  LockEx(int, std::recursive_mutex &m_) : m(m_) {
    m.lock();

  }

  ~LockEx() {

    m.unlock();
  }

private:
  std::recursive_mutex &m;
};

#define SPILOG(fn, fmt, ...) do {  \
      fn(fmt, ## __VA_ARGS__);     \
      fn("  %d / 0x%x / %d / %d / tx: %s", \
         xfer_count, header.endpoint, header.tx_len, header.max_rx_len, \
         util::hexdump(tx_buf, std::min((int)header.tx_len, 8)).c_str()); \
      } while (0)

PandaSpiHandle::PandaSpiHandle(std::string serial) {
  can_piggyback = util::getenv("SPI_CAN_PIGGYBACK", "1") == "1";
  int ret;
  const int uid_len = 12;
  uint8_t uid[uid_len] = {0};

  uint32_t spi_mode = SPI_MODE_0;
  uint8_t spi_bits_per_word = 8;

  // 50MHz is the max of the 845. note that some older
  // revs of the comma three may not support this speed
  uint32_t spi_speed = std::stoul(util::getenv("SPI_SPEED_HZ", "50000000"));
  try {
    if (!util::file_exists(SPI_DEVICE)) {
      throw std::runtime_error("Error connecting to panda: SPI device not found");
    }

    spi_fd = open(SPI_DEVICE.c_str(), O_RDWR);
    if (spi_fd < 0) {
      LOGE("failed opening SPI device %d", spi_fd);
      throw std::runtime_error("Error connecting to panda: failed to open SPI device");
    }

    // One owner for the connection lifetime, including discovery and retries.
    if (flock(spi_fd, LOCK_EX | LOCK_NB) != 0) {
      throw std::runtime_error("SPI device already owned");
    }

    // SPI settings
    util::safe_ioctl(spi_fd, SPI_IOC_WR_MODE, &spi_mode, "failed setting SPI mode");
    util::safe_ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed, "failed setting SPI speed");
    util::safe_ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &spi_bits_per_word, "failed setting SPI bits per word");

    int version = detect_protocol();
    if (version != 2 && version != 3) {
      throw std::runtime_error("Unsupported panda SPI protocol");
    }
    protocol_v3 = version == 3;
    if (protocol_v3) {
      if (getrandom(&session, sizeof(session), 0) != sizeof(session) || session == 0) {
        throw std::runtime_error("Could not generate SPI session");
      }
      if (spi_transfer_v3(0xfe, nullptr, 0, nullptr, 0, 500) < 0) {
        throw std::runtime_error("Could not initialize SPI session");
      }
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
    flock(spi_fd, LOCK_UN);
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
  LockEx lock(spi_fd, hw_lock);
  int ret = spi_transfer_retry(0, (uint8_t *) &packet, sizeof(packet), NULL, 0, timeout);
  return ret;
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

// A smallest CAN record occupies six bytes, so a full byte-sized batch can
// exceed a physical TX queue. Limit completed records as well as wire bytes.
static int can_write_chunk(const uint8_t *data, int length, uint8_t &remaining, int *completed = nullptr, int record_limit = 415) {
  static const uint8_t lengths[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64};
  const int limit = std::min(length, (int)SPI_PROTO_MAX_PAYLOAD);
  int pos = 0;
  int records = 0;
  // CAN_TX_BUFFER_SIZE is 416, with one ring slot kept empty. Counting all
  // buses together is conservative and avoids additional per-bus parser state.
  while (pos < limit && records < record_limit) {
    if (remaining == 0) remaining = 6 + lengths[data[pos] >> 4];
    int take = std::min((int)remaining, limit - pos);
    remaining -= take;
    pos += take;
    if (remaining == 0) ++records;
  }
  if (completed != nullptr) *completed = records;
  return pos;
}

int PandaSpiHandle::bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  std::lock_guard writer(write_lock);
  if (!protocol_v3 || endpoint != 3) return bulk_transfer(endpoint, data, length, NULL, 0, timeout);
  uint64_t generation;
  {
    LockEx lock(spi_fd, hw_lock);
    generation = can_write_generation;
  }
  int offset = 0;
  while (offset < length) {
    uint8_t remaining;
    {
      LockEx lock(spi_fd, hw_lock);
      if (generation != can_write_generation) {
        connected = comms_healthy = false;
        return -1;
      }
      remaining = can_write_remaining;
    }
    int size = can_write_chunk(data + offset, length - offset, remaining);
    int ret = spi_transfer_retry(endpoint, data + offset, size, nullptr, 0, timeout, &generation);
    if (ret < 0) {
      LockEx lock(spi_fd, hw_lock);
      if (ret == SpiError::NACK && (can_write_remaining != 0 || generation != can_write_generation)) {
        // The caller may discard this write after rejection. A retained prefix
        // must not consume bytes from its next, unrelated CAN write.
        connected = comms_healthy = false;
        return -1;
      }
      return ret;
    }
    {
      LockEx lock(spi_fd, hw_lock);
      if (generation != can_write_generation) {
        // A concurrent comms reset invalidated this byte stream. Never continue
        // its suffix or restore parser carry from before that reset.
        connected = comms_healthy = false;
        return -1;
      }
    }
    offset += size;
  }
  return 0;
}

int PandaSpiHandle::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return bulk_transfer(endpoint, NULL, 0, data, length, timeout);
}

int PandaSpiHandle::bulk_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t rx_len, unsigned int timeout) {
  const int xfer_size = protocol_v3 ? SPI_PROTO_MAX_PAYLOAD : 1984;

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
      if (!protocol_v3 || d != SpiError::NACK) comms_healthy = false;
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


int PandaSpiHandle::spi_transfer_retry(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout, const uint64_t *write_generation) {
  if (protocol_v3) return spi_transfer_v3(endpoint, tx_data, tx_len, rx_data, max_rx_len, timeout, write_generation);
  LockEx operation_lock(spi_fd, hw_lock);
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
      for (uint32_t i = 0; i < t.len; i++) {
        if ((static_cast<double>(rand()) / RAND_MAX) > 0.9) {
          ((uint8_t*)t.tx_buf)[i] = (uint8_t)(rand() % 256);
        }
      }
    }
  }

#ifdef SPI_STRESS_HOOKS
  int ret = stress_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t);
#else
  int ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t);
#endif

  if (err_prob > 0) {
    if ((static_cast<double>(rand()) / RAND_MAX) < err_prob && t.rx_buf != (uint64_t)NULL) {
      printf("corrupting RX\n");
      for (uint32_t i = 0; i < t.len; i++) {
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

  wait_for_spi_turnaround(spi_last_bus_activity_ns);

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
  wait_for_spi_turnaround(nanos_since_boot());

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

  spi_last_bus_activity_ns = nanos_since_boot();
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

  spi_last_bus_activity_ns = nanos_since_boot();
  if (ret >= 0) ret = -1;
  return ret;
}

// VERSION is deliberately shared with the bootstub's legacy protocol.
int PandaSpiHandle::detect_protocol() {
  LockEx lock(spi_fd, hw_lock);
  for (int attempt = 0; attempt < 5; ++attempt) {
    wait_for_spi_turnaround(spi_last_bus_activity_ns);
    memcpy(tx_buf, "VERSION", 7);
    spi_ioc_transfer request = {.tx_buf = (uint64_t)tx_buf, .len = 7};
    int sent = lltransfer(request);
    wait_for_spi_turnaround(nanos_since_boot());
    memset(tx_buf, 0, 25);
    memset(rx_buf, 0, 25);
    spi_ioc_transfer response = {.tx_buf = (uint64_t)tx_buf, .rx_buf = (uint64_t)rx_buf, .len = 25};
    int received = lltransfer(response);
    spi_last_bus_activity_ns = nanos_since_boot();
    if (sent != 7 || received != 25 || memcmp(rx_buf, "VERSION", 7) != 0 || rx_buf[7] != 15 || rx_buf[8] != 0) continue;
    uint8_t crc = 0xff;
    for (int i = 23; i >= 0; --i) {
      crc ^= rx_buf[i];
      for (int bit = 0; bit < 8; ++bit) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xd5) : (uint8_t)(crc << 1);
    }
    if (crc == rx_buf[24]) return rx_buf[23];
  }
  // Older bootstubs expose VERSION using the original header-poll/data-read
  // sequence. Keep this migration path separate: v3 ends TX at every CS rise.
  for (int attempt = 0; attempt < 5; ++attempt) {
    wait_for_spi_turnaround(spi_last_bus_activity_ns);
    memcpy(tx_buf, "VERSION", 7);
    spi_ioc_transfer request = {.tx_buf = (uint64_t)tx_buf, .len = 7};
    if (lltransfer(request) != 7) {
      spi_last_bus_activity_ns = nanos_since_boot();
      continue;
    }
    wait_for_spi_turnaround(nanos_since_boot());
    uint64_t deadline = nanos_since_boot() + 10000000;
    bool found = false;
    do {
      memset(tx_buf, 0, 9);
      memset(rx_buf, 0, 9);
      spi_ioc_transfer prefix = {.tx_buf = (uint64_t)tx_buf, .rx_buf = (uint64_t)rx_buf, .len = 9};
      int received = lltransfer(prefix);
      spi_last_bus_activity_ns = nanos_since_boot();
      found = received == 9 && memcmp(rx_buf, "VERSION", 7) == 0;
      if (found) break;
    } while (nanos_since_boot() < deadline);
    if (!found) continue;
    uint16_t length = rx_buf[7] | (rx_buf[8] << 8);
    if (length < 15 || length > 1000) continue;
    memset(tx_buf, 0, length + 1);
    spi_ioc_transfer payload = {.tx_buf = (uint64_t)tx_buf, .rx_buf = (uint64_t)(rx_buf + 9), .len = (uint32_t)length + 1};
    int received = lltransfer(payload);
    spi_last_bus_activity_ns = nanos_since_boot();
    if (received != length + 1) continue;
    uint8_t crc = 0xff;
    for (int i = 8 + length; i >= 0; --i) {
      crc ^= rx_buf[i];
      for (int bit = 0; bit < 8; ++bit) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xd5) : (uint8_t)(crc << 1);
    }
    if (crc == rx_buf[9 + length] && rx_buf[23] == 2) return 2;
  }
  return -1;
}

static void wait_for_v3_boundary(uint64_t start_ns, bool after_request, bool maintenance = false, int can_records = 0, bool can_read = false) {
  // Bench-derived gaps for the matched optimized firmware, including loaded
  // interrupt tests. These measurements do not establish an absolute timing bound.
  static const uint64_t request_gap = std::stoull(util::getenv("SPI_REQUEST_GAP_US", "500")) * 1000;
  static const uint64_t recovery_gap = std::stoull(util::getenv("SPI_RECOVERY_GAP_US", "250")) * 1000;
  static const uint64_t maintenance_gap = std::stoull(util::getenv("SPI_MAINTENANCE_GAP_US", "100000")) * 1000;
  static const uint64_t can_record_gap = std::stoull(util::getenv("SPI_CAN_RECORD_GAP_US", "4")) * 1000;
  static const uint64_t can_read_gap = std::stoull(util::getenv("SPI_CAN_READ_GAP_US", "750")) * 1000;
  const uint64_t preparation_gap = std::max(request_gap + can_records * can_record_gap, can_read ? can_read_gap : 0);
  const uint64_t gap = after_request ? (maintenance ? std::max(preparation_gap, maintenance_gap) : preparation_gap) : recovery_gap;
  while ((nanos_since_boot() - start_ns) < gap) {}
}

int PandaSpiHandle::spi_transfer_v3(uint8_t endpoint, const uint8_t *tx_data, uint16_t tx_len,
                                  uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout, const uint64_t *write_generation) {
  std::unique_lock<std::recursive_mutex> lock(hw_lock);
  if (!connected || !comms_healthy || tx_len > SPI_PROTO_MAX_PAYLOAD || max_rx_len > SPI_PROTO_MAX_PAYLOAD) return -1;
  if ((endpoint == 1 || endpoint == 0x81) && tx_len == 0 && can_rx_pending_size != 0) {
    uint16_t size = std::min(max_rx_len, can_rx_pending_size);
    if (rx_data != nullptr && size != 0) memcpy(rx_data, can_rx_pending + can_rx_pending_offset, size);
    can_rx_pending_offset += size;
    can_rx_pending_size -= size;
    if (can_rx_pending_size == 0) can_rx_pending_offset = 0;
    return size;  // A short read avoids mixing newer wire bytes with this prefix.
  }
  if (sequence == std::numeric_limits<uint32_t>::max()) {
    connected = comms_healthy = false;
    return -1;  // Reconnect before sequence wrap; never alias INIT.
  }
  const uint64_t operation_generation = write_generation != nullptr ? *write_generation : can_write_generation;
  const uint64_t deadline = nanos_since_boot() + uint64_t(timeout == 0 ? SPI_ACK_TIMEOUT : timeout) * 1000000;
  bool maintenance = endpoint == 2;
  if (endpoint == 0 && tx_len > 0) {
    switch (tx_data[0]) {
      case 0xb5: case 0xdc: case 0xde: case 0xe5: case 0xe7: case 0xf9: case 0xfc: maintenance = true; break;
      default: break;
    }
  }
  const uint32_t request_size = SPI_PROTO_OVERHEAD + tx_len;
  uint32_t response_size = SPI_PROTO_OVERHEAD + max_rx_len;
  std::array<uint8_t, SPI_PROTO_MAX_FRAME> request_bytes = {};
  std::array<uint8_t, SPI_PROTO_MAX_FRAME> dummy = {};
  spi_proto_header request_header = {
    .magic = SPI_PROTO_REQUEST, .endpoint = endpoint, .len = tx_len,
    .seq = sequence, .session = session, .capacity = max_rx_len,
    .version = SPI_PROTO_VERSION, .reserved = 0,
  };
  if (tx_len != 0) memcpy(request_bytes.data() + SPI_PROTO_HEADER_SIZE, tx_data, tx_len);
  int can_records = 0;
  uint8_t can_remaining_after = 0;
  auto encode_request = [&]() {
    if (endpoint == 3) {
      can_remaining_after = can_write_remaining;
      (void)can_write_chunk(tx_data, tx_len, can_remaining_after, &can_records, std::numeric_limits<int>::max());
    }
    // Capacity is frozen for this identity. Only a terminal rejection permits
    // recomputing it after readers have had a chance to drain the pending bytes.
    if (endpoint == 3) max_rx_len = can_piggyback && can_rx_pending_size == 0 ? SPI_PROTO_MAX_PAYLOAD : 0;
    request_header.capacity = max_rx_len;
    response_size = SPI_PROTO_OVERHEAD + max_rx_len;
    request_header.seq = sequence;
    memcpy(request_bytes.data(), &request_header, sizeof(request_header));
    uint32_t crc = spi_proto_crc32(request_bytes.data(), request_size - 4);
    memcpy(request_bytes.data() + request_size - 4, &crc, sizeof(crc));
  };
  encode_request();
  header = {.sync = SPI_SYNC, .endpoint = endpoint, .tx_len = tx_len, .max_rx_len = max_rx_len};
  bool outcome_unknown = false;
  do {
    wait_for_v3_boundary(spi_last_bus_activity_ns, false);
    if (nanos_since_boot() >= deadline) break;
    if ((endpoint == 3 || write_generation != nullptr) && operation_generation != can_write_generation) {
      connected = comms_healthy = false;
      return -1;
    }
    ++xfer_count;
#ifdef SPI_STRESS_HOOKS
    stress_attempt(endpoint, tx_len, max_rx_len);
    stress_phase(0);
#endif
    // lltransfer's fault injection may mutate its input. Preserve the retry bytes.
    memcpy(tx_buf, request_bytes.data(), request_size);
    spi_ioc_transfer request = {.tx_buf = (uint64_t)tx_buf, .len = request_size};
    static const double short_prob = std::stod(util::getenv("SPI_REQUEST_SHORT_PROB", "0"));
    if (endpoint != SPI_PROTO_INIT && short_prob > 0 && (double(rand()) / RAND_MAX) < short_prob) request.len = std::max(1U, request_size / 2);
    outcome_unknown = true;  // Even a failed ioctl may have executed the request.
    int sent = lltransfer(request);
    wait_for_v3_boundary(nanos_since_boot(), true, maintenance, can_records,
                        endpoint == 1 || endpoint == 0x81 || (endpoint == 3 && max_rx_len > 0));
#ifdef SPI_STRESS_HOOKS
    stress_phase(1);
#endif
    memset(rx_buf, 0, response_size);
    // Zero MOSI cannot be a REQUEST, including when a pair starts misaligned.
    dummy.fill(0);
    spi_ioc_transfer response = {.tx_buf = (uint64_t)dummy.data(), .rx_buf = (uint64_t)rx_buf, .len = response_size};
    int received = lltransfer(response);
    spi_last_bus_activity_ns = nanos_since_boot();
    static const double corrupt_prob = std::stod(util::getenv("SPI_RESPONSE_CORRUPT_PROB", "0"));
    if (endpoint != SPI_PROTO_INIT && corrupt_prob > 0 && (double(rand()) / RAND_MAX) < corrupt_prob) rx_buf[response_size - 1] ^= 1;
    // Even a failed ioctl can have executed the request: always complete the pair.
    if (sent != (int)request_size || received != (int)response_size) {
#ifdef SPI_STRESS_HOOKS
      stress_result(-1, false);
#endif
      continue;
    }
    uint32_t received_crc;
    memcpy(&received_crc, rx_buf + response_size - 4, sizeof(received_crc));
    if (received_crc != spi_proto_crc32(rx_buf, response_size - 4)) {
#ifdef SPI_STRESS_HOOKS
      stress_result(-1, false);
#endif
      continue;
    }
    spi_proto_header reply;
    memcpy(&reply, rx_buf, sizeof(reply));
    if (reply.magic != SPI_PROTO_RESPONSE || reply.version != SPI_PROTO_VERSION || reply.reserved != 0 ||
        reply.capacity != 0 || reply.len > max_rx_len || reply.session != session || reply.seq != sequence) {
#ifdef SPI_STRESS_HOOKS
      stress_result(-1, false);
#endif
      continue;
    }
#ifdef SPI_STRESS_HOOKS
    stress_result(reply.endpoint, true);
#endif
    if (reply.endpoint == SPI_PROTO_OK) {
      ++sequence;
      if (endpoint == SPI_PROTO_INIT || (endpoint == 0 && tx_len == sizeof(ControlPacket_t) && tx_data[0] == 0xc0)) {
        can_rx_pending_offset = can_rx_pending_size = 0;
        can_write_remaining = 0;
        ++can_write_generation;
      }
      if (endpoint == 3) {
        can_write_remaining = can_remaining_after;
        if (reply.len != 0) {
          assert(can_rx_pending_size == 0);
          memcpy(can_rx_pending, rx_buf + SPI_PROTO_HEADER_SIZE, reply.len);
          can_rx_pending_offset = 0;
          can_rx_pending_size = reply.len;
        }
        return 0;  // CAN write completion remains independent of piggyback bytes.
      }
      if (rx_data != nullptr && reply.len != 0) memcpy(rx_data, rx_buf + SPI_PROTO_HEADER_SIZE, reply.len);
      return reply.len;
    }
    if (reply.endpoint == SPI_PROTO_REJECTED && reply.len == 0) {
      // This is a completed zero-effect result. Only a NEW number may retry it.
      outcome_unknown = false;
      ++sequence;
      if (sequence == std::numeric_limits<uint32_t>::max()) break;
      if (endpoint != 3) return SpiError::NACK;
      // A terminal rejection resolves this operation. Let readers drain RX
      // receipts before allocating another operation; retain ownership flock.
      lock.unlock();
      usleep(200);
      lock.lock();
      if (!connected || !comms_healthy || sequence == std::numeric_limits<uint32_t>::max()) break;
      encode_request();
      header = {.sync = SPI_SYNC, .endpoint = endpoint, .tx_len = tx_len, .max_rx_len = max_rx_len};
      continue;
    }
    if (reply.endpoint == SPI_PROTO_SESSION || reply.endpoint == SPI_PROTO_SEQUENCE) {
      SPILOG(LOGE, "SPI v3: session or sequence lost, outcome unknown");
      connected = comms_healthy = false;
      return -1;
    }
  } while (connected && nanos_since_boot() < deadline);
  if (!outcome_unknown && connected && comms_healthy && sequence != std::numeric_limits<uint32_t>::max() &&
      (endpoint != 3 || can_write_remaining == 0) &&
      ((endpoint != 3 && write_generation == nullptr) || operation_generation == can_write_generation)) {
    return SpiError::NACK;  // No unresolved request: keep the session for readers and later writes.
  }
  SPILOG(LOGE, "SPI v3: deadline expired, ending session%s", outcome_unknown ? " (outcome unknown)" : "");
  connected = comms_healthy = false;
  return -1;
}
