#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>


#define TIMEOUT 0
#define SPI_BUF_SIZE 2048


class PandaSpiHandle {
public:
  std::string hw_serial;
  std::atomic<bool> connected = true;
  std::atomic<bool> comms_healthy = true;

  PandaSpiHandle(std::string serial);
  ~PandaSpiHandle();

  int control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout=TIMEOUT);
  int control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout=TIMEOUT);
  int bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  void cleanup();

  static std::vector<std::string> list();

private:
  int spi_fd = -1;
  uint8_t tx_buf[SPI_BUF_SIZE];
  uint8_t rx_buf[SPI_BUF_SIZE];
  inline static std::recursive_mutex hw_lock;

  struct __attribute__((packed)) spi_header {
    uint8_t sync;
    uint8_t endpoint;
    uint16_t tx_len;
    uint16_t max_rx_len;
  };

  int wait_for_ack(uint8_t ack, uint8_t tx, unsigned int timeout, unsigned int length);
  int bulk_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t rx_len, unsigned int timeout);
  int spi_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout);
  int spi_transfer_retry(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len, unsigned int timeout);
  int lltransfer(struct spi_ioc_transfer &t);

  spi_header header;
  uint32_t xfer_count = 0;
};
