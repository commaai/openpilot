#pragma once

#define SPI_BUF_SIZE 1024U
#define SPI_TIMEOUT_US 10000U

#ifdef STM32H7
__attribute__((section(".ram_d1"))) uint8_t spi_buf_rx[SPI_BUF_SIZE];
__attribute__((section(".ram_d1"))) uint8_t spi_buf_tx[SPI_BUF_SIZE];
#else
uint8_t spi_buf_rx[SPI_BUF_SIZE];
uint8_t spi_buf_tx[SPI_BUF_SIZE];
#endif

#define SPI_CHECKSUM_START 0xABU
#define SPI_SYNC_BYTE 0x5AU
#define SPI_HACK 0x79U
#define SPI_DACK 0x85U
#define SPI_NACK 0x1FU

// SPI states
enum {
  SPI_STATE_HEADER,
  SPI_STATE_HEADER_ACK,
  SPI_STATE_HEADER_NACK,
  SPI_STATE_DATA_RX,
  SPI_STATE_DATA_RX_ACK,
  SPI_STATE_DATA_TX
};

uint8_t spi_state = SPI_STATE_HEADER;
uint8_t spi_endpoint;
uint16_t spi_data_len_mosi;
uint16_t spi_data_len_miso;

#define SPI_HEADER_SIZE 7U

// low level SPI prototypes
void llspi_init(void);
void llspi_mosi_dma(uint8_t *addr, int len);
void llspi_miso_dma(uint8_t *addr, int len);

void spi_init(void) {
  // clear buffers (for debugging)
  (void)memset(spi_buf_rx, 0, SPI_BUF_SIZE);
  (void)memset(spi_buf_tx, 0, SPI_BUF_SIZE);

  // platform init
  llspi_init();

  // Start the first packet!
  spi_state = SPI_STATE_HEADER;
  llspi_mosi_dma(spi_buf_rx, SPI_HEADER_SIZE);
}

bool check_checksum(uint8_t *data, uint16_t len) {
  // TODO: can speed this up by casting the bulk to uint32_t and xor-ing the bytes afterwards
  uint8_t checksum = SPI_CHECKSUM_START;
  for(uint16_t i = 0U; i < len; i++){
    checksum ^= data[i];
  }
  return checksum == 0U;
}

void spi_handle_rx(void) {
  uint8_t next_rx_state = SPI_STATE_HEADER;

  // parse header
  spi_endpoint = spi_buf_rx[1];
  spi_data_len_mosi = (spi_buf_rx[3] << 8) | spi_buf_rx[2];
  spi_data_len_miso = (spi_buf_rx[5] << 8) | spi_buf_rx[4];

  if (spi_state == SPI_STATE_HEADER) {
    if ((spi_buf_rx[0] == SPI_SYNC_BYTE) && check_checksum(spi_buf_rx, SPI_HEADER_SIZE)) {
      // response: ACK and start receiving data portion
      spi_buf_tx[0] = SPI_HACK;
      next_rx_state = SPI_STATE_HEADER_ACK;
    } else {
      // response: NACK and reset state machine
      print("- incorrect header sync or checksum "); hexdump(spi_buf_rx, SPI_HEADER_SIZE);
      spi_buf_tx[0] = SPI_NACK;
      next_rx_state = SPI_STATE_HEADER_NACK;
    }
    llspi_miso_dma(spi_buf_tx, 1);
  } else if (spi_state == SPI_STATE_DATA_RX) {
    // We got everything! Based on the endpoint specified, call the appropriate handler
    uint16_t response_len = 0U;
    bool reponse_ack = false;
    if (check_checksum(&(spi_buf_rx[SPI_HEADER_SIZE]), spi_data_len_mosi + 1U)) {
      if (spi_endpoint == 0U) {
        if (spi_data_len_mosi >= sizeof(ControlPacket_t)) {
          ControlPacket_t ctrl;
          (void)memcpy(&ctrl, &spi_buf_rx[SPI_HEADER_SIZE], sizeof(ControlPacket_t));
          response_len = comms_control_handler(&ctrl, &spi_buf_tx[3]);
          reponse_ack = true;
        } else {
          print("SPI: insufficient data for control handler\n");
        }
      } else if ((spi_endpoint == 1U) || (spi_endpoint == 0x81U)) {
        if (spi_data_len_mosi == 0U) {
          response_len = comms_can_read(&(spi_buf_tx[3]), spi_data_len_miso);
          reponse_ack = true;
        } else {
          print("SPI: did not expect data for can_read\n");
        }
      } else if (spi_endpoint == 2U) {
        comms_endpoint2_write(&spi_buf_rx[SPI_HEADER_SIZE], spi_data_len_mosi);
        reponse_ack = true;
      } else if (spi_endpoint == 3U) {
        if (spi_data_len_mosi > 0U) {
          comms_can_write(&spi_buf_rx[SPI_HEADER_SIZE], spi_data_len_mosi);
          reponse_ack = true;
        } else {
          print("SPI: did expect data for can_write\n");
        }
      } else {
        print("SPI: unexpected endpoint"); puth(spi_endpoint); print("\n");
      }
    } else {
      // Checksum was incorrect
      reponse_ack = false;
      print("- incorrect data checksum\n");
    }

    // Setup response header
    spi_buf_tx[0] = reponse_ack ? SPI_DACK : SPI_NACK;
    spi_buf_tx[1] = response_len & 0xFFU;
    spi_buf_tx[2] = (response_len >> 8) & 0xFFU;

    // Add checksum
    uint8_t checksum = SPI_CHECKSUM_START;
    for(uint16_t i = 0U; i < (response_len + 3U); i++) {
      checksum ^= spi_buf_tx[i];
    }
    spi_buf_tx[response_len + 3U] = checksum;

    // Write response
    llspi_miso_dma(spi_buf_tx, response_len + 4U);

    next_rx_state = SPI_STATE_DATA_TX;
  } else {
    print("SPI: RX unexpected state: "); puth(spi_state); print("\n");
  }

  spi_state = next_rx_state;
}

void spi_handle_tx(bool timed_out) {
  if ((spi_state == SPI_STATE_HEADER_NACK) || timed_out) {
    // Reset state
    spi_state = SPI_STATE_HEADER;
    llspi_mosi_dma(spi_buf_rx, SPI_HEADER_SIZE);
  } else if (spi_state == SPI_STATE_HEADER_ACK) {
    // ACK was sent, queue up the RX buf for the data + checksum
    spi_state = SPI_STATE_DATA_RX;
    llspi_mosi_dma(&spi_buf_rx[SPI_HEADER_SIZE], spi_data_len_mosi + 1U);
  } else if (spi_state == SPI_STATE_DATA_TX) {
    // Reset state
    spi_state = SPI_STATE_HEADER;
    llspi_mosi_dma(spi_buf_rx, SPI_HEADER_SIZE);
  } else {
    spi_state = SPI_STATE_HEADER;
    llspi_mosi_dma(spi_buf_rx, SPI_HEADER_SIZE);
    print("SPI: TX unexpected state: "); puth(spi_state); print("\n");
  }
}
