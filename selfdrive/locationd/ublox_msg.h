#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <ctime>

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "selfdrive/locationd/generated/gps.h"
#include "selfdrive/locationd/generated/ubx.h"

using namespace std::string_literals;

// protocol constants
namespace ublox {
  const uint8_t PREAMBLE1 = 0xb5;
  const uint8_t PREAMBLE2 = 0x62;

  const int UBLOX_HEADER_SIZE = 6;
  const int UBLOX_CHECKSUM_SIZE = 2;
  const int UBLOX_MAX_MSG_SIZE = 65536;

  struct ubx_mga_ini_time_utc_t {
    uint8_t type;
    uint8_t version;
    uint8_t ref;
    int8_t leapSecs;
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
    uint8_t reserved1;
    uint32_t ns;
    uint16_t tAccS;
    uint16_t reserved2;
    uint32_t tAccNs;
  } __attribute__((packed));

  inline std::string ubx_add_checksum(const std::string &msg) {
    assert(msg.size() > 2);

    uint8_t ck_a = 0, ck_b = 0;
    for(int i = 2; i < msg.size(); i++) {
      ck_a = (ck_a + msg[i]) & 0xFF;
      ck_b = (ck_b + ck_a) & 0xFF;
    }

    std::string r = msg;
    r.push_back(ck_a);
    r.push_back(ck_b);
    return r;
  }

  inline std::string build_ubx_mga_ini_time_utc(struct tm time) {
    ublox::ubx_mga_ini_time_utc_t payload = {
      .type = 0x10,
      .version = 0x0,
      .ref = 0x0,
      .leapSecs = -128, // Unknown
      .year = (uint16_t)(1900 + time.tm_year),
      .month = (uint8_t)(1 + time.tm_mon),
      .day = (uint8_t)time.tm_mday,
      .hour = (uint8_t)time.tm_hour,
      .minute = (uint8_t)time.tm_min,
      .second = (uint8_t)time.tm_sec,
      .reserved1 = 0x0,
      .ns = 0,
      .tAccS = 30,
      .reserved2 = 0x0,
      .tAccNs = 0,
    };
    assert(sizeof(payload) == 24);

    std::string msg = "\xb5\x62\x13\x40\x18\x00"s;
    msg += std::string((char*)&payload, sizeof(payload));

    return ubx_add_checksum(msg);
  }
}

class UbloxMsgParser {
  public:
    bool add_data(const uint8_t *incoming_data, uint32_t incoming_data_len, size_t &bytes_consumed);
    inline void reset() {bytes_in_parse_buf = 0;}
    inline int needed_bytes();
    inline std::string data() {return std::string((const char*)msg_parse_buf, bytes_in_parse_buf);}

    std::pair<std::string, kj::Array<capnp::word>> gen_msg();
    kj::Array<capnp::word> gen_nav_pvt(ubx_t::nav_pvt_t *msg);
    kj::Array<capnp::word> gen_rxm_sfrbx(ubx_t::rxm_sfrbx_t *msg);
    kj::Array<capnp::word> gen_rxm_rawx(ubx_t::rxm_rawx_t *msg);
    kj::Array<capnp::word> gen_mon_hw(ubx_t::mon_hw_t *msg);
    kj::Array<capnp::word> gen_mon_hw2(ubx_t::mon_hw2_t *msg);

  private:
    inline bool valid_cheksum();
    inline bool valid();
    inline bool valid_so_far();

    kj::Array<capnp::word> parse_gps_ephemeris(ubx_t::rxm_sfrbx_t *msg);

    std::unordered_map<int, std::unordered_map<int, std::string>> gps_subframes;

    size_t bytes_in_parse_buf = 0;
    uint8_t msg_parse_buf[ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_MAX_MSG_SIZE];

};

