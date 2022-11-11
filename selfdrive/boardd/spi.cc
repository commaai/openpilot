#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

#include <cassert>
#include <cstring>

#include "common/util.h"
#include "common/swaglog.h"
#include "panda/board/comms_definitions.h"
#include "selfdrive/boardd/panda_comms.h"


#define SPI_SYNC 0x5AU
#define SPI_HACK 0x79U
#define SPI_DACK 0x85U
#define SPI_NACK 0x1FU
#define SPI_CHECKSUM_START 0xABU

struct __attribute__((packed)) spi_header {
  uint8_t sync;
  uint8_t endpoint;
  uint16_t tx_len;
  uint16_t max_rx_len;
};


PandaSpiHandle::PandaSpiHandle(std::string serial) : PandaCommsHandle(serial) {
  LOGD("opening SPI panda: %s", serial.c_str());

  int err;
  uint32_t spi_mode = SPI_MODE_0;
  uint32_t spi_speed = 30000000;
  uint8_t spi_bits_per_word = 8;

  spi_fd = open(serial.c_str(), O_RDWR);
  if (spi_fd < 0) {
    LOGE("failed opening SPI device %d", err);
    goto fail;
  }

  // SPI settings
  err = util::safe_ioctl(spi_fd, SPI_IOC_WR_MODE, &spi_mode);
  if (err < 0) {
    LOGE("failed setting SPI mode %d", err);
    goto fail;
  }

  err = util::safe_ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed);
  if (err < 0) {
    LOGE("failed setting SPI speed");
    goto fail;
  }

  err = util::safe_ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &spi_bits_per_word);
  if (err < 0) {
    LOGE("failed setting SPI bits per word");
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
  return spi_transfer_retry(0, (uint8_t *) &packet, sizeof(packet), NULL, 0);
}

int PandaSpiHandle::control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout) {
  ControlPacket_t packet = {
    .request = request,
    .param1 = param1,
    .param2 = param2,
    .length = length
  };
  return spi_transfer_retry(0, (uint8_t *) &packet, sizeof(packet), data, length);
}

int PandaSpiHandle::bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return 0;
}

int PandaSpiHandle::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  return 0;
}

std::vector<std::string> PandaSpiHandle::list() {
  // TODO: list all pandas available over SPI
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


int PandaSpiHandle::spi_transfer_retry(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len) {
  int err;

  std::lock_guard lk(hw_lock);
  do {
    // TODO: handle error
    err = spi_transfer(endpoint, tx_data, tx_len, rx_data, max_rx_len);
  } while (err < 0 && connected && !PANDA_NO_RETRY);

  return err;
}

int PandaSpiHandle::wait_for_ack(spi_ioc_transfer &transfer, uint8_t ack) {
  // TODO: add timeout?
  while (true) {
    int ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
    if (ret < 0) {
      LOGE("SPI: failed to send ACK request");
      return ret;
    }

    if (rx_buf[0] == ack) {
      break;
    } else if (rx_buf[0] == SPI_NACK) {
      LOGW("SPI: got header NACK");
      return -1;
    }
  }

  return 0;
}

int PandaSpiHandle::spi_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len) {
  int ret;
  uint16_t rx_data_len;

  // needs to be less, since we need to have space for the checksum
  assert(tx_len < SPI_BUF_SIZE);
  assert(max_rx_len < SPI_BUF_SIZE);

  spi_header header = {
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
  ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) {
    LOGE("SPI: failed to send header");
    goto transfer_fail;
  }

  // Wait for (N)ACK
  tx_buf[0] = 0x12;
  transfer.len = 1;
  ret = wait_for_ack(transfer, SPI_HACK);
  if (ret < 0) {
    goto transfer_fail;
  }

  // Send data
  if (tx_data != NULL) {
    memcpy(tx_buf, tx_data, tx_len);
  }
  add_checksum(tx_buf, tx_len);
  transfer.len = tx_len + 1;
  ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) {
    LOGE("SPI: failed to send data");
    goto transfer_fail;
  }

  // Wait for (N)ACK
  tx_buf[0] = 0xab;
  transfer.len = 1;
  ret = wait_for_ack(transfer, SPI_DACK);
  if (ret < 0) {
    goto transfer_fail;
  }

  // Read data len
  transfer.len = 2;
  transfer.rx_buf = (uint64_t)(rx_buf + 1);
  ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) {
    LOGE("SPI: failed to read rx data len");
    goto transfer_fail;
  }
  rx_data_len = *(uint16_t *)(rx_buf+1);
  assert(rx_data_len < SPI_BUF_SIZE);

  // Read data
  transfer.len = rx_data_len + 1;
  transfer.rx_buf = (uint64_t)(rx_buf + 2 + 1);
  ret = util::safe_ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) {
    LOGE("SPI: failed to read rx data");
    goto transfer_fail;
  }
  if (!check_checksum(rx_buf, rx_data_len + 4)) {
    LOGE("SPI: bad checksum");
    goto transfer_fail;
  }

  if (rx_data != NULL) {
    memcpy(rx_data, rx_buf + 3, rx_data_len);
  }

  return rx_data_len;

transfer_fail:
  return ret;
}
