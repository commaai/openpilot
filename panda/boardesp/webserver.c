#include "stdlib.h"
#include "ets_sys.h"
#include "osapi.h"
#include "gpio.h"
#include "mem.h"
#include "os_type.h"
#include "user_interface.h"
#include "espconn.h"
#include "upgrade.h"

#include "crypto/rsa.h"
#include "crypto/sha.h"

#include "obj/gitversion.h"
#include "obj/cert.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define espconn_send_string(conn, x) espconn_send(conn, x, strlen(x))

#define MAX_RESP 0x800
char resp[MAX_RESP];
char pageheader[] = "HTTP/1.0 200 OK\nContent-Type: text/html\n\n"
"<!DOCTYPE html>\n"
"<html>\n"
"<head>\n"
"<title>Panda</title>\n"
"</head>\n"
"<body>\n"
"<pre>This is your comma.ai panda\n\n"
"It's open source. Find the code <a href=\"https://github.com/commaai/panda\">here</a>\n"
"Designed to work with our dashcam, <a href=\"http://chffr.comma.ai\">chffr</a>\n";

char pagefooter[] = "</pre>\n"
"</body>\n"
"</html>\n";

char OK_header[] = "HTTP/1.0 200 OK\nContent-Type: text/html\n\n";

static struct espconn web_conn;
static esp_tcp web_proto;
extern char ssid[];
extern int wifi_secure_mode;

char *st_firmware;
int real_content_length, content_length = 0;
char *st_firmware_ptr;
LOCAL os_timer_t ota_reboot_timer;

#define FIRMWARE_SIZE 503808

typedef struct {
  uint16_t ep;
  uint16_t extra_len;
  union {
    struct {
      uint8_t request_type;
      uint8_t request;
      uint16_t value;
      uint16_t index;
      uint16_t length;
    } control;
    uint8_t data[0x10];
  } u;
} usb_msg;

int ICACHE_FLASH_ATTR usb_cmd(int ep, int len, int request,
                              int value, int index, char *data) {
  usb_msg usb = {0};

  usb.ep = ep;
  usb.extra_len = (ep == 0) ? 0 : len;
  if (ep == 0) {
    usb.u.control.request_type = 0xc0;
    usb.u.control.request = request;
    usb.u.control.value = value;
    usb.u.control.index = index;
  } else {
    memcpy(&usb.u.data, data, usb.extra_len);
  }

  uint32_t recv[0x44/4];
  spi_comm(&usb, sizeof(usb), recv, 0x40);

  return recv[0];
}
 

void ICACHE_FLASH_ATTR st_flash() {
  if (st_firmware != NULL) {
    // boot mode
    os_printf("st_flash: enter boot mode\n");
    st_set_boot_mode(1);

    // echo
    os_printf("st_flash: wait for echo\n");
    for (int i = 0; i < 10; i++) {
      os_printf("  attempt: %d\n", i);
      if (usb_cmd(0, 0, 0xb0, 0, 0, NULL) > 0) break;
    }

    // unlock flash
    os_printf("st_flash: unlock flash\n");
    usb_cmd(0, 0, 0xb1, 0, 0, NULL);

    // erase sector 1
    os_printf("st_flash: erase sector 1\n");
    usb_cmd(0, 0, 0xb2, 1, 0, NULL);

    if (real_content_length >= 16384) {
      // erase sector 2
      os_printf("st_flash: erase sector 2\n");
      usb_cmd(0, 0, 0xb2, 2, 0, NULL);
    }

    // real content length will always be 0x10 aligned
    os_printf("st_flash: flashing\n");
    for (int i = 0; i < real_content_length; i += 0x10) {
      int rl = MIN(0x10, real_content_length-i);
      usb_cmd(2, rl, 0, 0, 0, &st_firmware[i]);
      system_soft_wdt_feed();
    }

    // reboot into normal mode
    os_printf("st_flash: rebooting\n");
    usb_cmd(0, 0, 0xd8, 0, 0, NULL);

    // done with this
    os_free(st_firmware);
    st_firmware = NULL;
  }
}

typedef enum {
    NOT_STARTED,
    CONNECTION_ESTABLISHED,
    RECEIVING_HEADER,
    RECEIVING_ST_FIRMWARE,
    RECEIVING_ESP_FIRMWARE,
    REBOOTING,
    ERROR
} web_state_t;

web_state_t state = NOT_STARTED;
int esp_address, esp_address_erase_limit, start_address;

void ICACHE_FLASH_ATTR hexdump(char *data, int len) {
  int i;
  for (i=0;i<len;i++) {
    if (i!=0 && (i%0x10)==0) os_printf("\n");
    os_printf("%02X ", data[i]);
  }
  os_printf("\n");
}

void ICACHE_FLASH_ATTR st_reset() {
  // reset the ST
  gpio16_output_conf();
  gpio16_output_set(0);
  os_delay_us(1000);
  gpio16_output_set(1);
  os_delay_us(10000);
}

void ICACHE_FLASH_ATTR st_set_boot_mode(int boot_mode) {
  if (boot_mode) {
    // boot mode (pull low)
    gpio_output_set(0, (1 << 4), (1 << 4), 0);
    st_reset();
  } else {
    // no boot mode (pull high)
    gpio_output_set((1 << 4), 0, (1 << 4), 0);
    st_reset();
  }

  // float boot pin
  gpio_output_set(0, 0, 0, (1 << 4));
}

static void ICACHE_FLASH_ATTR web_rx_cb(void *arg, char *data, uint16_t len) {
  int i;
  struct espconn *conn = (struct espconn *)arg;
  if (state == CONNECTION_ESTABLISHED) {
    state = RECEIVING_HEADER;
    os_printf("%s %d\n", data, len);

    // index
    if (memcmp(data, "GET / ", 6) == 0) {
      memset(resp, 0, MAX_RESP);

      strcpy(resp, pageheader);
      ets_strcat(resp, "\nssid: ");
      ets_strcat(resp, ssid);
      ets_strcat(resp, "\n");

      ets_strcat(resp, "\nst version:     ");
      uint32_t recvData[0x11];
      int len = spi_comm("\x00\x00\x00\x00\x40\xD6\x00\x00\x00\x00\x40\x00", 0xC, recvData, 0x40);
      ets_memcpy(resp+strlen(resp), recvData+1, len);

      ets_strcat(resp, "\nesp version:    ");
      ets_strcat(resp, gitversion);
      uint8_t current = system_upgrade_userbin_check();
      if (current == UPGRADE_FW_BIN1) {
        ets_strcat(resp, "\nesp flash file: user2.bin");
      } else {
        ets_strcat(resp, "\nesp flash file: user1.bin");
      }

      if (wifi_secure_mode) {
        ets_strcat(resp, "\nin secure mode");
      } else {
        ets_strcat(resp, "\nin INSECURE mode...<a href=\"/secure\">secure it</a>");
      }
      
      ets_strcat(resp,"\nSet USB Mode:"
        "<button onclick=\"var xhr = new XMLHttpRequest(); xhr.open('GET', 'set_property?usb_mode=0'); xhr.send()\" type='button'>Client</button>"
        "<button onclick=\"var xhr = new XMLHttpRequest(); xhr.open('GET', 'set_property?usb_mode=1'); xhr.send()\" type='button'>CDP</button>"
        "<button onclick=\"var xhr = new XMLHttpRequest(); xhr.open('GET', 'set_property?usb_mode=2'); xhr.send()\" type='button'>DCP</button>\n");

      ets_strcat(resp, pagefooter);
      
      espconn_send_string(&web_conn, resp);
      espconn_disconnect(conn);
    } else if (memcmp(data, "GET /secure", 11) == 0 && !wifi_secure_mode) {
      wifi_configure(1);
    } else if (memcmp(data, "GET /set_property?usb_mode=", 27) == 0 && wifi_secure_mode) {
      char mode_value = data[27] - '0';
        if (mode_value >= '\x00' && mode_value <= '\x02') {
          memset(resp, 0, MAX_RESP);
          char set_usb_mode_packet[] = "\x00\x00\x00\x00\x40\xE6\x00\x00\x00\x00\x40\x00";
          set_usb_mode_packet[6] = mode_value;
          uint32_t recvData[1];
          spi_comm(set_usb_mode_packet, 0xC, recvData, 0);
          os_sprintf(resp, "%sUSB Mode set to %02x\n\n", OK_header, mode_value);
          espconn_send_string(&web_conn, resp);
          espconn_disconnect(conn);
        }  
    } else if (memcmp(data, "PUT /stupdate ", 14) == 0 && wifi_secure_mode) {
      os_printf("init st firmware\n");
      char *cl = strstr(data, "Content-Length: ");
      if (cl != NULL) {
        // get content length
        cl += strlen("Content-Length: ");
        content_length = skip_atoi(&cl);
        os_printf("with content length %d\n", content_length);

        // should be small enough to fit in RAM
        real_content_length = (content_length+0xF)&(~0xF);
        st_firmware_ptr = st_firmware = os_malloc(real_content_length);
        memset(st_firmware, 0, real_content_length);
        state = RECEIVING_ST_FIRMWARE;
      }
      
    } else if (((memcmp(data, "PUT /espupdate1 ", 16) == 0) ||
                (memcmp(data, "PUT /espupdate2 ", 16) == 0)) && wifi_secure_mode) {
      // 0x1000   = user1.bin
      // 0x81000  = user2.bin
      // 0x3FE000 = blank.bin
      os_printf("init esp firmware\n");
      char *cl = strstr(data, "Content-Length: ");
      if (cl != NULL) {
        // get content length
        cl += strlen("Content-Length: ");
        content_length = skip_atoi(&cl);
        os_printf("with content length %d\n", content_length);

        // setup flashing
        uint8_t current = system_upgrade_userbin_check();
        if (data[14] == '2' && current == UPGRADE_FW_BIN1) {
          os_printf("flashing boot2.bin\n");
          state = RECEIVING_ESP_FIRMWARE;
          esp_address = 4*1024 + FIRMWARE_SIZE + 16*1024 + 4*1024;
        } else if (data[14] == '1' && current == UPGRADE_FW_BIN2) {
          os_printf("flashing boot1.bin\n");
          state = RECEIVING_ESP_FIRMWARE;
          esp_address = 4*1024;
        } else {
          espconn_send_string(&web_conn, "HTTP/1.0 404 Not Found\nContent-Type: text/html\n\nwrong!\n");
          espconn_disconnect(conn);
        }
        esp_address_erase_limit = esp_address;
        start_address = esp_address;
      }
    } else {
      espconn_send_string(&web_conn, "HTTP/1.0 404 Not Found\nContent-Type: text/html\n\n404 Not Found!\n");
      espconn_disconnect(conn);
    }
  } else if (state == RECEIVING_ST_FIRMWARE) {
    os_printf("receiving st firmware: %d/%d\n", len, content_length);
    memcpy(st_firmware_ptr, data, MIN(content_length, len));
    st_firmware_ptr += len;
    content_length -= len;

    if (content_length <= 0 && real_content_length > 1000) {
      state = NOT_STARTED;
      os_printf("done!\n");
      espconn_send_string(&web_conn, "HTTP/1.0 200 OK\nContent-Type: text/html\n\nsuccess!\n");
      espconn_disconnect(conn);

      // reboot
      os_printf("Scheduling st_flash in 100ms.\n");
      os_timer_disarm(&ota_reboot_timer);
      os_timer_setfn(&ota_reboot_timer, (os_timer_func_t *)st_flash, NULL);
      os_timer_arm(&ota_reboot_timer, 100, 0);
    }
  } else if (state == RECEIVING_ESP_FIRMWARE) {
    if ((esp_address+len) < (start_address + FIRMWARE_SIZE)) {
      os_printf("receiving esp firmware: %d/%d -- 0x%x - 0x%x\n", len, content_length,
        esp_address, esp_address_erase_limit);
      content_length -= len;
      while (esp_address_erase_limit < (esp_address + len)) {
        os_printf("erasing 0x%X\n", esp_address_erase_limit);
        spi_flash_erase_sector(esp_address_erase_limit / SPI_FLASH_SEC_SIZE);
        esp_address_erase_limit += SPI_FLASH_SEC_SIZE;
      }
      SpiFlashOpResult res = spi_flash_write(esp_address, data, len);
      if (res != SPI_FLASH_RESULT_OK) {
        os_printf("flash fail @ 0x%x\n", esp_address);
      }
      esp_address += len;

      if (content_length == 0) {

        char digest[SHA_DIGEST_SIZE];
        uint32_t rsa[RSANUMBYTES/4];
        uint32_t dat[0x80/4];
        int ll;
        spi_flash_read(esp_address-RSANUMBYTES, rsa, RSANUMBYTES);

        // 32-bit aligned accesses only
        SHA_CTX ctx;
        SHA_init(&ctx);
        for (ll = start_address; ll < esp_address-RSANUMBYTES; ll += 0x80) {
          spi_flash_read(ll, dat, 0x80);
          SHA_update(&ctx, dat, MIN((esp_address-RSANUMBYTES)-ll, 0x80));
        }
        memcpy(digest, SHA_final(&ctx), SHA_DIGEST_SIZE);

        if (RSA_verify(&releaseesp_rsa_key, rsa, RSANUMBYTES, digest, SHA_DIGEST_SIZE) ||
          #ifdef ALLOW_DEBUG
            RSA_verify(&debugesp_rsa_key, rsa, RSANUMBYTES, digest, SHA_DIGEST_SIZE)
          #else
            false
          #endif
          ) {
          os_printf("RSA verify success!\n");
          espconn_send_string(&web_conn, "HTTP/1.0 200 OK\nContent-Type: text/html\n\nsuccess!\n");
          system_upgrade_flag_set(UPGRADE_FLAG_FINISH);

          // reboot
          os_printf("Scheduling reboot.\n");
          os_timer_disarm(&ota_reboot_timer);
          os_timer_setfn(&ota_reboot_timer, (os_timer_func_t *)system_upgrade_reboot, NULL);
          os_timer_arm(&ota_reboot_timer, 2000, 0);
        } else {
          os_printf("RSA verify FAILURE\n");
          espconn_send_string(&web_conn, "HTTP/1.0 500 Internal Server Error\nContent-Type: text/html\n\nrsa verify fail\n");
        }
        espconn_disconnect(conn);
      }
    }
  }
}

void ICACHE_FLASH_ATTR web_tcp_connect_cb(void *arg) {
  state = CONNECTION_ESTABLISHED;
  struct espconn *conn = (struct espconn *)arg;
  espconn_set_opt(&web_conn, ESPCONN_NODELAY);
  espconn_regist_recvcb(conn, web_rx_cb);
}

void ICACHE_FLASH_ATTR web_init() {
  web_proto.local_port = 80;
  web_conn.type = ESPCONN_TCP;
  web_conn.state = ESPCONN_NONE;
  web_conn.proto.tcp = &web_proto;
  espconn_regist_connectcb(&web_conn, web_tcp_connect_cb);
  espconn_accept(&web_conn);
}

