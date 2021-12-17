#include "selfdrive/boardd/pigeon.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <optional>

#include "selfdrive/common/gpio.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/locationd/ublox_msg.h"

// Termios on macos doesn't define all baud rate constants
#ifndef B460800
#define   B460800 0010004
#endif

using namespace std::string_literals;

extern ExitHandler do_exit;

const std::string ack = "\xb5\x62\x05\x01\x02\x00";
const std::string nack = "\xb5\x62\x05\x00\x02\x00";
const std::string sos_save_ack = "\xb5\x62\x09\x14\x08\x00\x02\x00\x00\x00\x01\x00\x00\x00";
const std::string sos_save_nack = "\xb5\x62\x09\x14\x08\x00\x02\x00\x00\x00\x00\x00\x00\x00";

Pigeon * Pigeon::connect(Panda * p) {
  PandaPigeon * pigeon = new PandaPigeon();
  pigeon->connect(p);

  return pigeon;
}

Pigeon * Pigeon::connect(const char * tty) {
  TTYPigeon * pigeon = new TTYPigeon();
  pigeon->connect(tty);

  return pigeon;
}

bool Pigeon::wait_for_ack(const std::string &ack_, const std::string &nack_, int timeout_ms) {
  std::string s;
  const double start_t = millis_since_boot();
  while (!do_exit) {
    s += receive();

    if (s.find(ack_) != std::string::npos) {
      LOGD("Received ACK from ublox");
      return true;
    } else if (s.find(nack_) != std::string::npos) {
      LOGE("Received NACK from ublox");
      return false;
    } else if (s.size() > 0x1000 || ((millis_since_boot() - start_t) > timeout_ms)) {
      LOGE("No response from ublox");
      return false;
    }

    util::sleep_for(1); // Allow other threads to be scheduled
  }
  return false;
}

bool Pigeon::wait_for_ack() {
  return wait_for_ack(ack, nack);
}

bool Pigeon::send_with_ack(const std::string &cmd) {
  send(cmd);
  return wait_for_ack();
}

sos_restore_response Pigeon::wait_for_backup_restore_status(int timeout_ms) {
  std::string s;
  const double start_t = millis_since_boot();
  while (!do_exit) {
    s += receive();

    size_t position = s.find("\xb5\x62\x09\x14\x08\x00\x03");
    if (position != std::string::npos && s.size() >= (position + 11)) {
      return static_cast<sos_restore_response>(s[position + 10]);
    } else if (s.size() > 0x1000 || ((millis_since_boot() - start_t) > timeout_ms)) {
      LOGE("No backup restore response from ublox");
      return error;
    }

    util::sleep_for(1); // Allow other threads to be scheduled
  }
  return error;
}

void Pigeon::init() {
  for (int i = 0; i < 10; i++) {
    if (do_exit) return;
    LOGW("panda GPS start");

    // power off pigeon
    set_power(false);
    util::sleep_for(100);

    // 9600 baud at init
    set_baud(9600);

    // power on pigeon
    set_power(true);
    util::sleep_for(500);

    // baud rate upping
    send("\x24\x50\x55\x42\x58\x2C\x34\x31\x2C\x31\x2C\x30\x30\x30\x37\x2C\x30\x30\x30\x33\x2C\x34\x36\x30\x38\x30\x30\x2C\x30\x2A\x31\x35\x0D\x0A"s);
    util::sleep_for(100);

    // set baud rate to 460800
    set_baud(460800);

    // init from ubloxd
    // To generate this data, run selfdrive/locationd/test/ubloxd.py
    if (!send_with_ack("\xB5\x62\x06\x00\x14\x00\x03\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x1E\x7F"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x00\x14\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x35"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x00\x14\x00\x01\x00\x00\x00\xC0\x08\x00\x00\x00\x08\x07\x00\x01\x00\x01\x00\x00\x00\x00\x00\xF4\x80"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x00\x14\x00\x04\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1D\x85"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x00\x00\x00\x06\x18"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x00\x01\x00\x01\x08\x22"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x00\x01\x00\x03\x0A\x24"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x08\x06\x00\x64\x00\x01\x00\x00\x00\x79\x10"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x24\x24\x00\x05\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5A\x63"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x1E\x14\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3C\x37"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x39\x08\x00\xFF\xAD\x62\xAD\x1E\x63\x00\x00\x83\x0C"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x24\x00\x00\x2A\x84"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x23\x00\x00\x29\x81"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x1E\x00\x00\x24\x72"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x39\x00\x00\x3F\xC3"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x01\x03\x00\x01\x07\x01\x13\x51"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x01\x03\x00\x02\x15\x01\x22\x70"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x01\x03\x00\x02\x13\x01\x20\x6C"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x01\x03\x00\x0A\x09\x01\x1E\x70"s)) continue;
    if (!send_with_ack("\xB5\x62\x06\x01\x03\x00\x0A\x0B\x01\x20\x74"s)) continue;

    // check the backup restore status
    send("\xB5\x62\x09\x14\x00\x00\x1D\x60"s);
    sos_restore_response restore_status = wait_for_backup_restore_status();
    switch(restore_status) {
      case restored:
        LOGW("almanac backup restored");
        // clear the backup
        send_with_ack("\xB5\x62\x06\x01\x03\x00\x0A\x0B\x01\x20\x74"s);
        break;
      case no_backup:
        LOGW("no almanac backup found");
        break;
      default:
        LOGE("failed to restore almanac backup, status: %d", restore_status);
    }

    auto time = util::get_time();
    if (util::time_valid(time)) {
      LOGW("Sending current time to ublox");
      send(ublox::build_ubx_mga_ini_time_utc(time));
    }

    LOGW("panda GPS on");
    return;
  }
  LOGE("failed to initialize panda GPS");
}

void Pigeon::stop() {
  LOGW("Storing almanac in ublox flash");

  // Controlled GNSS stop
  send("\xB5\x62\x06\x04\x04\x00\x00\x00\x08\x00\x16\x74"s);

  // Store almanac in flash
  send("\xB5\x62\x09\x14\x04\x00\x00\x00\x00\x00\x21\xEC"s);

  if (wait_for_ack(sos_save_ack, sos_save_nack)) {
    LOGW("Done storing almanac");
  } else {
    LOGE("Error storing almanac");
  }
}

void PandaPigeon::connect(Panda * p) {
  panda = p;
}

void PandaPigeon::set_baud(int baud) {
  panda->write(0xe2, 1, 0);
  panda->write(0xe4, 1, baud/300);
}

void PandaPigeon::send(const std::string &s) {
  int len = s.length();
  const char * dat = s.data();

  unsigned char a[0x20+1];
  a[0] = 1;
  for (int i=0; i<len; i+=0x20) {
    int ll = std::min(0x20, len-i);
    memcpy(&a[1], &dat[i], ll);

    panda->bulk_write(2, a, ll+1);
  }
}

std::string PandaPigeon::receive() {
  std::string r;
  r.reserve(0x1000 + 0x40);
  unsigned char dat[0x40];
  while (r.length() < 0x1000) {
    int len = panda->read(0xe0, 1, 0, dat, sizeof(dat));
    if (len <= 0) break;
    r.append((char*)dat, len);
  }

  return r;
}

void PandaPigeon::set_power(bool power) {
  panda->write(0xd9, power, 0);
}

PandaPigeon::~PandaPigeon() {
}

void handle_tty_issue(int err, const char func[]) {
  LOGE_100("tty error %d \"%s\" in %s", err, strerror(err), func);
}

void TTYPigeon::connect(const char * tty) {
  pigeon_tty_fd = HANDLE_EINTR(open(tty, O_RDWR));
  if (pigeon_tty_fd < 0) {
    handle_tty_issue(errno, __func__);
    assert(pigeon_tty_fd >= 0);
  }
  int err = tcgetattr(pigeon_tty_fd, &pigeon_tty);
  assert(err == 0);

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

  err = tcsetattr(pigeon_tty_fd, TCSANOW, &pigeon_tty);
  assert(err == 0);
}

void TTYPigeon::set_baud(int baud) {
  speed_t baud_const = 0;
  switch(baud) {
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
  int err = tcdrain(pigeon_tty_fd);
  assert(err == 0);

  // change baud
  err = tcgetattr(pigeon_tty_fd, &pigeon_tty);
  assert(err == 0);
  err = cfsetspeed(&pigeon_tty, baud_const);
  assert(err == 0);
  err = tcsetattr(pigeon_tty_fd, TCSANOW, &pigeon_tty);
  assert(err == 0);

  // flush
  err = tcflush(pigeon_tty_fd, TCIOFLUSH);
  assert(err == 0);
}

void TTYPigeon::send(const std::string &s) {
  int err = HANDLE_EINTR(write(pigeon_tty_fd, s.data(), s.length()));

  if(err < 0) { handle_tty_issue(err, __func__); }
  err = tcdrain(pigeon_tty_fd);
  if(err < 0) { handle_tty_issue(err, __func__); }
}

std::string TTYPigeon::receive() {
  std::string r;
  r.reserve(0x1000 + 0x40);
  char dat[0x40];
  while (r.length() < 0x1000) {
    int len = read(pigeon_tty_fd, dat, sizeof(dat));
    if(len < 0) {
      handle_tty_issue(len, __func__);
    } else if (len == 0) {
      break;
    } else {
      r.append(dat, len);
    }

  }
  return r;
}

void TTYPigeon::set_power(bool power) {
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

TTYPigeon::~TTYPigeon() {
  close(pigeon_tty_fd);
}
