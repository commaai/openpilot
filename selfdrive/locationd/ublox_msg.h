#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "messaging.hpp"
#include "generated/ubx.h"
#include "generated/gps.h"

// protocol constants
namespace ublox {
  const uint8_t PREAMBLE1 = 0xb5;
  const uint8_t PREAMBLE2 = 0x62;

  const int UBLOX_HEADER_SIZE = 6;
  const int UBLOX_CHECKSUM_SIZE = 2;
  const int UBLOX_MAX_MSG_SIZE = 65536;

  // Boardd still uses these:
  const uint8_t CLASS_NAV = 0x01;
  const uint8_t CLASS_RXM = 0x02;
  const uint8_t CLASS_MON = 0x0A;
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

    std::unordered_map<int, std::unordered_map<int, std::string>> gps_subframes;

    size_t bytes_in_parse_buf = 0;
    uint8_t msg_parse_buf[ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_MAX_MSG_SIZE];

};

