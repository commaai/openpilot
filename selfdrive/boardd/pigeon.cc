#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>

#include "common/swaglog.h"
#include "common/gpio.h"

#include "pigeon.h"

// Termios on macos doesn't define all baud rate constants
#ifndef B460800
#define   B460800 0010004
#endif

using namespace std::string_literals;


Pigeon * Pigeon::connect(Panda * p){
  PandaPigeon * pigeon = new PandaPigeon();
  pigeon->connect(p);

  return pigeon;
}

Pigeon * Pigeon::connect(const char * tty){
  TTYPigeon * pigeon = new TTYPigeon();
  pigeon->connect(tty);

  return pigeon;
}

void Pigeon::init() {
  usleep(1000*1000);
  LOGW("panda GPS start");

  // power off pigeon
  set_power(0);
  usleep(100*1000);

  // 9600 baud at init
  set_baud(9600);

  // power on pigeon
  set_power(1);
  usleep(500*1000);

  // baud rate upping
  send("\x24\x50\x55\x42\x58\x2C\x34\x31\x2C\x31\x2C\x30\x30\x30\x37\x2C\x30\x30\x30\x33\x2C\x34\x36\x30\x38\x30\x30\x2C\x30\x2A\x31\x35\x0D\x0A"s);
  usleep(100*1000);

  // set baud rate to 460800
  set_baud(460800);
  usleep(100*1000);

  // init from ubloxd
  // To generate this data, run test/ubloxd.py with the print statements enabled in the write function in panda/python/serial.py
  send("\xB5\x62\x06\x00\x14\x00\x03\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x1E\x7F"s);
  send("\xB5\x62\x06\x3E\x00\x00\x44\xD2"s);
  send("\xB5\x62\x06\x00\x14\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x35"s);
  send("\xB5\x62\x06\x00\x14\x00\x01\x00\x00\x00\xC0\x08\x00\x00\x00\x08\x07\x00\x01\x00\x01\x00\x00\x00\x00\x00\xF4\x80"s);
  send("\xB5\x62\x06\x00\x14\x00\x04\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1D\x85"s);
  send("\xB5\x62\x06\x00\x00\x00\x06\x18"s);
  send("\xB5\x62\x06\x00\x01\x00\x01\x08\x22"s);
  send("\xB5\x62\x06\x00\x01\x00\x02\x09\x23"s);
  send("\xB5\x62\x06\x00\x01\x00\x03\x0A\x24"s);
  send("\xB5\x62\x06\x08\x06\x00\x64\x00\x01\x00\x00\x00\x79\x10"s);
  send("\xB5\x62\x06\x24\x24\x00\x05\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5A\x63"s);
  send("\xB5\x62\x06\x1E\x14\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3C\x37"s);
  send("\xB5\x62\x06\x24\x00\x00\x2A\x84"s);
  send("\xB5\x62\x06\x23\x00\x00\x29\x81"s);
  send("\xB5\x62\x06\x1E\x00\x00\x24\x72"s);
  send("\xB5\x62\x06\x01\x03\x00\x01\x07\x01\x13\x51"s);
  send("\xB5\x62\x06\x01\x03\x00\x02\x15\x01\x22\x70"s);
  send("\xB5\x62\x06\x01\x03\x00\x02\x13\x01\x20\x6C"s);
  send("\xB5\x62\x06\x01\x03\x00\x0A\x09\x01\x1E\x70"s);

  LOGW("panda GPS on");
}

void PandaPigeon::connect(Panda * p) {
  panda = p;
}

void PandaPigeon::set_baud(int baud) {
  panda->usb_write(0xe2, 1, 0);
  panda->usb_write(0xe4, 1, baud/300);
}

void PandaPigeon::send(std::string s) {
  int len = s.length();
  const char * dat = s.data();

  unsigned char a[0x20+1];
  a[0] = 1;
  for (int i=0; i<len; i+=0x20) {
    int ll = std::min(0x20, len-i);
    memcpy(&a[1], &dat[i], ll);

    panda->usb_bulk_write(2, a, ll+1);
  }
}

std::string PandaPigeon::receive() {
  std::string r;

  while (true){
    unsigned char dat[0x40];
    int len = panda->usb_read(0xe0, 1, 0, dat, sizeof(dat));
    if (len <= 0 || r.length() > 0x1000) break;
    r.append((char*)dat, len);
  }

  return r;
}

void PandaPigeon::set_power(bool power) {
  panda->usb_write(0xd9, power, 0);
}

PandaPigeon::~PandaPigeon(){
}

void handle_tty_issue(int err, const char func[]) {
  LOGE_100("tty error %d \"%s\" in %s", err, strerror(err), func);
}

void TTYPigeon::connect(const char * tty) {
  pigeon_tty_fd = open(tty, O_RDWR);
  if (pigeon_tty_fd < 0){
    handle_tty_issue(errno, __func__);
    assert(pigeon_tty_fd >= 0);
  }
  assert(tcgetattr(pigeon_tty_fd, &pigeon_tty) == 0);

  // configure tty
  pigeon_tty.c_cflag &= ~PARENB;                                            // disable parity
  pigeon_tty.c_cflag &= ~CSTOPB;                                            // single stop bit
  pigeon_tty.c_cflag |= CS8;                                                // 8 bits per byte
  pigeon_tty.c_cflag &= ~CRTSCTS;                                           // no RTS/CTS flow control
  pigeon_tty.c_cflag |= CREAD | CLOCAL;                                     // turn on READ & ignore ctrl lines
  pigeon_tty.c_lflag &= ~ICANON;                                            // disable canonical mode
  pigeon_tty.c_lflag &= ~ISIG;                                              // disable interpretation of INTR, QUIT and SUSP
  pigeon_tty.c_iflag &= ~(IXON | IXOFF | IXANY);                            // turn off software flow ctrl
  pigeon_tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);   // disable any special handling of received bytes
  pigeon_tty.c_oflag &= ~OPOST;                                             // prevent special interpretation of output bytes
  pigeon_tty.c_oflag &= ~ONLCR;                                             // prevent conversion of newline to carriage return/line feed

  // configure blocking behavior
  pigeon_tty.c_cc[VMIN] = 0;  // min amount of characters returned
  pigeon_tty.c_cc[VTIME] = 0; // max blocking time in s/10 (0=inf)

  assert(tcsetattr(pigeon_tty_fd, TCSANOW, &pigeon_tty) == 0);
}

void TTYPigeon::set_baud(int baud){
  speed_t baud_const = 0;
  switch(baud){
  case 9600:
    baud_const = B9600;
    break;
  case 460800:
    baud_const = B460800;
    break;
  default:
    assert(false);
  }

  // make sure everything is tx'ed before changing baud
  assert(tcdrain(pigeon_tty_fd) == 0);

  // change baud
  assert(tcgetattr(pigeon_tty_fd, &pigeon_tty) == 0);
  assert(cfsetspeed(&pigeon_tty, baud_const) == 0);
  assert(tcsetattr(pigeon_tty_fd, TCSANOW, &pigeon_tty) == 0);

  // flush
  assert(tcflush(pigeon_tty_fd, TCIOFLUSH) == 0);
}

void TTYPigeon::send(std::string s) {
  int len = s.length();
  const char * dat = s.data();

  int err = write(pigeon_tty_fd, dat, len);
  if(err < 0) { handle_tty_issue(err, __func__); }
  err = tcdrain(pigeon_tty_fd);
  if(err < 0) { handle_tty_issue(err, __func__); }
}

std::string TTYPigeon::receive() {
  std::string r;

  while (true){
    char dat[0x40];
    int len = read(pigeon_tty_fd, dat, sizeof(dat));
    if(len < 0) {
      handle_tty_issue(len, __func__);
    } else if (len == 0 || r.length() > 0x1000){
      break;
    } else {
      r.append(dat, len);
    }

  }
  return r;
}

void TTYPigeon::set_power(bool power){
#ifdef QCOM2
  int err = 0;
  err += gpio_init(GPIO_UBLOX_RST_N, true);
  err += gpio_init(GPIO_UBLOX_SAFEBOOT_N, true);
  err += gpio_init(GPIO_UBLOX_PWR_EN, true);

  err += gpio_set(GPIO_UBLOX_RST_N, power);
  err += gpio_set(GPIO_UBLOX_SAFEBOOT_N, power);
  err += gpio_set(GPIO_UBLOX_PWR_EN, power);
  assert(err == 0);
#endif
}

TTYPigeon::~TTYPigeon(){
  close(pigeon_tty_fd);
}
