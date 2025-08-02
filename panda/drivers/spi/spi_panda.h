#include <linux/delay.h>
#include <linux/spi/spi.h>
#include <linux/spi/spidev.h>

#define SPI_SYNC 0x5AU
#define SPI_HACK 0x79U
#define SPI_DACK 0x85U
#define SPI_NACK 0x1FU
#define SPI_CHECKSUM_START 0xABU

struct __attribute__((packed)) spi_header {
  u8 sync;
  u8 endpoint;
  uint16_t tx_len;
  uint16_t max_rx_len;
};

struct spi_panda_transfer {
  __u64 rx_buf;
  __u64 tx_buf;
  __u32 tx_length;
  __u32 rx_length_max;
  __u32 timeout;
  __u8 endpoint;
  __u8 expect_disconnect;
};

static u8 panda_calc_checksum(u8 *buf, u16 length) {
  int i;
  u8 checksum = SPI_CHECKSUM_START;
  for (i = 0U; i < length; i++) {
    checksum ^= buf[i];
  }
  return checksum;
}

static long panda_wait_for_ack(struct spidev_data *spidev, u8 ack_val, u8 length) {
  int i;
  int ret;
  for (i = 0; i < 1000; i++) {
    ret = spidev_sync_read(spidev, length);
    if (ret < 0) {
      return ret;
    }

    if (spidev->rx_buffer[0] == ack_val) {
      return 0;
    } else if (spidev->rx_buffer[0] == SPI_NACK) {
      return -2;
    }
    if (i > 20) usleep_range(10, 20);
  }
  return -1;
}

static long panda_transfer_raw(struct spidev_data *spidev, struct spi_device *spi, unsigned long arg) {
  u16 rx_len;
  long retval = -1;
  struct spi_header header;
  struct spi_panda_transfer pt;

  struct spi_transfer t = {
    .len = 0,
    .tx_buf = spidev->tx_buffer,
    .rx_buf = spidev->rx_buffer,
    .speed_hz = spidev->spi->max_speed_hz,
  };

  struct spi_message m;
  spi_message_init(&m);
  spi_message_add_tail(&t, &m);

  // read struct from user
  if (!access_ok(VERIFY_WRITE, arg, sizeof(pt))) {
    return -1;
  }
  if (copy_from_user(&pt, (void __user *)arg, sizeof(pt))) {
    return -1;
  }
  dev_dbg(&spi->dev, "ep: %d, tx len: %d\n", pt.endpoint, pt.tx_length);

  // send header
  header.sync = 0x5a;
  header.endpoint = pt.endpoint;
  header.tx_len = pt.tx_length;
  header.max_rx_len = pt.rx_length_max;
  memcpy(spidev->tx_buffer, &header, sizeof(header));
  spidev->tx_buffer[sizeof(header)] = panda_calc_checksum(spidev->tx_buffer, sizeof(header));

  t.len = sizeof(header) + 1;
  retval = spidev_sync(spidev, &m);
  if (retval < 0) {
    dev_dbg(&spi->dev, "spi xfer failed %ld\n", retval);
    return retval;
  }

  // wait for ACK
  retval = panda_wait_for_ack(spidev, SPI_HACK, 1);
  if (retval < 0) {
    dev_dbg(&spi->dev, "no header ack %ld\n", retval);
    return retval;
  }

  // send data
  dev_dbg(&spi->dev, "sending data\n");
  retval = copy_from_user(spidev->tx_buffer, (const u8 __user *)(uintptr_t)pt.tx_buf, pt.tx_length);
  spidev->tx_buffer[pt.tx_length] = panda_calc_checksum(spidev->tx_buffer, pt.tx_length);
  t.len = pt.tx_length + 1;
  retval = spidev_sync(spidev, &m);

  if (pt.expect_disconnect) {
    return 0;
  }

  // wait for ACK
  retval = panda_wait_for_ack(spidev, SPI_DACK, 3);
  if (retval < 0) {
    dev_dbg(&spi->dev, "no data ack\n");
    return retval;
  }

  // get response
  t.rx_buf = spidev->rx_buffer + 3;
  rx_len = (spidev->rx_buffer[2] << 8) | (spidev->rx_buffer[1]);
  dev_dbg(&spi->dev, "rx len %u\n", rx_len);
  if (rx_len > pt.rx_length_max) {
    dev_dbg(&spi->dev, "RX len greater than max\n");
    return -1;
  }

  // do the read
  t.len = rx_len + 1;
  retval = spidev_sync(spidev, &m);
  if (retval < 0) {
    dev_dbg(&spi->dev, "spi xfer failed %ld\n", retval);
    return retval;
  }
  if (panda_calc_checksum(spidev->rx_buffer, 3 + rx_len + 1) != 0) {
    dev_dbg(&spi->dev, "bad checksum\n");
    return -1;
  }

  retval = copy_to_user((u8 __user *)(uintptr_t)pt.rx_buf, spidev->rx_buffer + 3, rx_len);

  return rx_len;
}

static long panda_transfer(struct spidev_data *spidev, struct spi_device *spi, unsigned long arg) {
  int i;
  int ret;
  dev_dbg(&spi->dev, "=== XFER start ===\n");
  for (i = 0; i < 20; i++) {
    ret = panda_transfer_raw(spidev, spi, arg);
    if (ret >= 0) {
      break;
    }
  }
  dev_dbg(&spi->dev, "took %d tries\n", i+1);
  return ret;
}
