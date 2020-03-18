#include "ets_sys.h"
#include "osapi.h"
#include "gpio.h"
#include "os_type.h"
#include "user_interface.h"
#include "espconn.h"

#include "driver/spi_interface.h"
#include "driver/uart.h"
#include "crypto/sha.h"

#define MIN(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

#define MAX(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

char ssid[32];
char password[] = "testing123";
int wifi_secure_mode = 0;

static const int pin = 2;

// Structure holding the TCP connection information.
struct espconn tcp_conn;
// TCP specific protocol structure.
esp_tcp tcp_proto;

// interrupt communication on port 1338, UDP!
struct espconn inter_conn;
esp_udp inter_proto;

uint32_t sendData[0x14] = {0};
uint32_t recvData[0x40] = {0};

static int ICACHE_FLASH_ATTR __spi_comm(char *dat, int len, uint32_t *recvData, int recvDataLen) {
  unsigned int length = 0;

  SpiData spiData;

  spiData.cmd = 2;
  spiData.cmdLen = 0;
  spiData.addr = NULL;
  spiData.addrLen = 0;

  // float boot pin
  gpio_output_set(0, 0, 0, (1 << 4));

  // manual CS pin
  gpio_output_set(0, (1 << 5), 0, 0);
  memset(sendData, 0xCC, 0x14);

  // wait for ST to respond to CS interrupt
  os_delay_us(50);

  // send request
  memcpy(((void*)sendData), dat, len);
  spiData.data = sendData;
  spiData.dataLen = 0x14;
  SPIMasterSendData(SpiNum_HSPI, &spiData);

  #define SPI_TIMEOUT 50000
  // give the ST time to be ready, up to 500ms
  int i;
  for (i = 0; (gpio_input_get() & (1 << 4)) && i < SPI_TIMEOUT; i++) {
    os_delay_us(10);
    system_soft_wdt_feed();
  }

  // TODO: handle this better
  if (i == SPI_TIMEOUT) {
    os_printf("ERROR: SPI receive failed\n");
    goto fail;
  }

  // blank out recvData
  memset(recvData, 0x00, 0x44);

  // receive the length
  spiData.data = recvData;
  spiData.dataLen = 4;
  if(SPIMasterRecvData(SpiNum_HSPI, &spiData) == -1) {
    // TODO: Handle gracefully. Maybe fail if len read fails?
    os_printf("SPI: Failed to recv length\n");
    goto fail;
  }

  length = recvData[0];
  if (length > 0x40) {
    os_printf("SPI: BAD LENGTH RECEIVED %x\n", length);
    length = 0;
    goto fail;
  }

  // got response, 0x40 works, 0x44 does not
  spiData.data = recvData+1;
  spiData.dataLen = (length+3)&(~3); // recvDataLen;
  if(SPIMasterRecvData(SpiNum_HSPI, &spiData) == -1) {
    // TODO: Handle gracefully. Maybe retry if payload failed.
    os_printf("SPI: Failed to recv payload\n");
    length = 0;
    goto fail;
  }

fail:
  // clear CS
  gpio_output_set((1 << 5), 0, 0, 0);

  // set boot pin back
  gpio_output_set((1 << 4), 0, (1 << 4), 0);

  return length;
}

int ICACHE_FLASH_ATTR spi_comm(char *dat, int len, uint32_t *recvData, int recvDataLen) {
  // blink the led during SPI comm
  if (GPIO_REG_READ(GPIO_OUT_ADDRESS) & (1 << pin)) {
    // set gpio low
    gpio_output_set(0, (1 << pin), 0, 0);
  } else {
    // set gpio high
    gpio_output_set((1 << pin), 0, 0, 0);
  }

  return __spi_comm(dat, len, recvData, recvDataLen);
}

static void ICACHE_FLASH_ATTR tcp_rx_cb(void *arg, char *data, uint16_t len) {
  // nothing too big
  if (len > 0x14) return;

  // do the SPI comm
  spi_comm(data, len, recvData, 0x40);

  espconn_send(&tcp_conn, recvData, 0x44);
}

void ICACHE_FLASH_ATTR tcp_connect_cb(void *arg) {
  struct espconn *conn = (struct espconn *)arg;
  espconn_set_opt(&tcp_conn, ESPCONN_NODELAY);
  espconn_regist_recvcb(conn, tcp_rx_cb);
}

// must be 0x44, because we can fit 4 more
uint8_t buf[0x44*0x10];
int queue_send_len = -1;

void ICACHE_FLASH_ATTR poll_can(void *arg) {
  uint8_t timerRecvData[0x44] = {0};
  int i = 0;
  int j;

  while (i < 0x40) {
    int len = spi_comm("\x01\x00\x00\x00", 4, timerRecvData, 0x40);
    if (len == 0) break;
    if (len > 0x40) { os_printf("SPI LENGTH ERROR!"); break; }

    // if it sends it, assume it's valid CAN
    for (j = 0; j < len; j += 0x10) {
      memcpy(buf + i*0x10, (timerRecvData+4)+j, 0x10);
      i++;
    }
  }

  if (i != 0) {
    int ret = espconn_sendto(&inter_conn, buf, i*0x10);
    if (ret != 0) {
			os_printf("send failed: %d\n", ret);
      queue_send_len = i*0x10;
    } else {
      queue_send_len = -1;
    }
  }
}

int udp_countdown = 0;

static volatile os_timer_t udp_callback;
void ICACHE_FLASH_ATTR udp_callback_func(void *arg) {
  if (queue_send_len == -1) {
    poll_can(NULL);
  } else {
    int ret = espconn_sendto(&inter_conn, buf, queue_send_len);
    if (ret == 0) {
      queue_send_len = -1;
    }
  }
  if (udp_countdown > 0) {
    os_timer_arm(&udp_callback, 5, 0);
    udp_countdown--;
  } else {
    os_printf("UDP timeout\n");
  }
}

void ICACHE_FLASH_ATTR inter_recv_cb(void *arg, char *pusrdata, unsigned short length) {
  remot_info *premot = NULL;
  if (espconn_get_connection_info(&inter_conn,&premot,0) == ESPCONN_OK) {
		inter_conn.proto.udp->remote_port = premot->remote_port;
		inter_conn.proto.udp->remote_ip[0] = premot->remote_ip[0];
		inter_conn.proto.udp->remote_ip[1] = premot->remote_ip[1];
		inter_conn.proto.udp->remote_ip[2] = premot->remote_ip[2];
		inter_conn.proto.udp->remote_ip[3] = premot->remote_ip[3];


    if (udp_countdown == 0) {
      os_printf("UDP recv\n");
      udp_countdown = 200*5;

      // start 5 second timer
      os_timer_disarm(&udp_callback);
      os_timer_setfn(&udp_callback, (os_timer_func_t *)udp_callback_func, NULL);
      os_timer_arm(&udp_callback, 5, 0);
    } else {
      udp_countdown = 200*5;
    }
  }
}

void ICACHE_FLASH_ATTR wifi_configure(int secure) {
  wifi_secure_mode = secure;

  // start wifi AP
  wifi_set_opmode(SOFTAP_MODE);
  struct softap_config config = {0};
  wifi_softap_get_config(&config);
  strcpy(config.ssid, ssid);
  if (wifi_secure_mode == 0) strcat(config.ssid, "-pair");
  strcpy(config.password, password);
  config.ssid_len = strlen(config.ssid);
  config.authmode = wifi_secure_mode ? AUTH_WPA2_PSK : AUTH_OPEN;
  config.beacon_interval = 100;
  config.max_connection = 4;
  wifi_softap_set_config(&config);

  if (wifi_secure_mode) {
    // setup tcp server
    tcp_proto.local_port = 1337;
    tcp_conn.type = ESPCONN_TCP;
    tcp_conn.state = ESPCONN_NONE;
    tcp_conn.proto.tcp = &tcp_proto;
    espconn_regist_connectcb(&tcp_conn, tcp_connect_cb);
    espconn_accept(&tcp_conn);
    espconn_regist_time(&tcp_conn, 60, 0); // 60s timeout for all connections

    // setup inter server
    inter_proto.local_port = 1338;
    const char udp_remote_ip[4] = {255, 255, 255, 255};
    os_memcpy(inter_proto.remote_ip, udp_remote_ip, 4);
    inter_proto.remote_port = 1338;

    inter_conn.type = ESPCONN_UDP;
    inter_conn.proto.udp = &inter_proto;

    espconn_regist_recvcb(&inter_conn, inter_recv_cb);
    espconn_create(&inter_conn);
  }
}

void ICACHE_FLASH_ATTR wifi_init() {
  // default ssid and password
  memset(ssid, 0, 32);
  os_sprintf(ssid, "panda-%08x-BROKEN", system_get_chip_id());

  // fetch secure ssid and password
  // update, try 20 times, for 1 second
  for (int i = 0; i < 20; i++) {
    uint8_t digest[SHA_DIGEST_SIZE];
    char resp[0x20];
    __spi_comm("\x00\x00\x00\x00\x40\xD0\x00\x00\x00\x00\x20\x00", 0xC, recvData, 0x40);
    memcpy(resp, recvData+1, 0x20);

    SHA_hash(resp, 0x1C, digest);
    if (memcmp(digest, resp+0x1C, 4) == 0) {
      // OTP is valid
      memcpy(ssid+6, resp, 0x10);
      memcpy(password, resp+0x10, 10);
      break;
    }
    os_delay_us(50000);
  }
  os_printf("Finished getting SID\n");
  os_printf(ssid);
  os_printf("\n");

  // set IP
  wifi_softap_dhcps_stop(); //stop DHCP before setting static IP
  struct ip_info ip_config;
  IP4_ADDR(&ip_config.ip, 192, 168, 0, 10);
  IP4_ADDR(&ip_config.gw, 0, 0, 0, 0);
  IP4_ADDR(&ip_config.netmask, 255, 255, 255, 0);
  wifi_set_ip_info(SOFTAP_IF, &ip_config);
  int stupid_gateway = 0;
  wifi_softap_set_dhcps_offer_option(OFFER_ROUTER, &stupid_gateway);
  wifi_softap_dhcps_start();

  wifi_configure(0);
}

#define LOOP_PRIO 2
#define QUEUE_SIZE 1
static os_event_t my_queue[QUEUE_SIZE];
void loop();

void ICACHE_FLASH_ATTR web_init();
void ICACHE_FLASH_ATTR elm327_init();

void ICACHE_FLASH_ATTR user_init() {
  // init gpio subsystem
  gpio_init();

  // configure UART TXD to be GPIO1, set as output
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_U0TXD_U, FUNC_GPIO1);
  gpio_output_set(0, 0, (1 << pin), 0);

  // configure SPI
  SpiAttr hSpiAttr;
  hSpiAttr.bitOrder = SpiBitOrder_MSBFirst;
  hSpiAttr.speed = SpiSpeed_10MHz;
  hSpiAttr.mode = SpiMode_Master;
  hSpiAttr.subMode = SpiSubMode_0;

  // TODO: is one of these CS?
  WRITE_PERI_REG(PERIPHS_IO_MUX, 0x105);
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_MTDI_U, 2);  // configure io to spi mode
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_MTCK_U, 2);  // configure io to spi mode
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_MTMS_U, 2);  // configure io to spi mode
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_MTDO_U, 2);  // configure io to spi mode
  SPIInit(SpiNum_HSPI, &hSpiAttr);
  //SPICsPinSelect(SpiNum_HSPI, SpiPinCS_1);

  // configure UART TXD to be GPIO1, set as output
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_GPIO5_U, FUNC_GPIO5);
  gpio_output_set(0, 0, (1 << 5), 0);
  gpio_output_set((1 << 5), 0, 0, 0);

  // uart init
  uart_init(BIT_RATE_115200, BIT_RATE_115200);

  // led init
  PIN_FUNC_SELECT(PERIPHS_IO_MUX_GPIO2_U, FUNC_GPIO2);
  gpio_output_set(0, (1 << pin), (1 << pin), 0);

  os_printf("hello\n");

  // needs SPI
  wifi_init();

  // support ota upgrades
  elm327_init();
  web_init();

  // set gpio high, so LED is off by default
  for (int i = 0; i < 5; i++) {
    gpio_output_set(0, (1 << pin), 0, 0);
    os_delay_us(50000);
    gpio_output_set((1 << pin), 0, 0, 0);
    os_delay_us(50000);
  }

  // jump to OS
  system_os_task(loop, LOOP_PRIO, my_queue, QUEUE_SIZE);
  system_os_post(LOOP_PRIO, 0, 0);
}

void ICACHE_FLASH_ATTR loop(os_event_t *events) {
  system_os_post(LOOP_PRIO, 0, 0);
}

