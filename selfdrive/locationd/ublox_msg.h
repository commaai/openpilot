#pragma once

#include <stdint.h>
#include "messaging.hpp"

#define min(x, y) ((x) <= (y) ? (x) : (y))

// NAV_PVT
typedef struct __attribute__((packed)) {
  uint32_t iTOW;
  uint16_t year;
  int8_t month;
  int8_t day;
  int8_t hour;
  int8_t min;
  int8_t sec;
  int8_t valid;
  uint32_t tAcc;
  int32_t nano;
  int8_t fixType;
  int8_t flags;
  int8_t flags2;
  int8_t numSV;
  int32_t lon;
  int32_t lat;
  int32_t height;
  int32_t hMSL;
  uint32_t hAcc;
  uint32_t vAcc;
  int32_t velN;
  int32_t velE;
  int32_t velD;
  int32_t gSpeed;
  int32_t headMot;
  uint32_t sAcc;
  uint32_t headAcc;
  uint16_t pDOP;
  int8_t reserverd1[6];
  int32_t headVeh;
  int16_t magDec;
  uint16_t magAcc;
} nav_pvt_msg;

// RXM_RAW
typedef struct __attribute__((packed)) {
  double rcvTow;
  uint16_t week;
  int8_t leapS;
  int8_t numMeas;
  int8_t recStat;
  int8_t reserved1[3];
} rxm_raw_msg;

// Extra data count is in numMeas
typedef struct __attribute__((packed)) {
  double prMes;
  double cpMes;
  float doMes;
  int8_t gnssId;
  int8_t svId;
  int8_t sigId;
  int8_t freqId;
  uint16_t locktime;
  int8_t cno;
  int8_t prStdev;
  int8_t cpStdev;
  int8_t doStdev;
  int8_t trkStat;
  int8_t reserved3;
} rxm_raw_msg_extra;
// RXM_SFRBX
typedef struct __attribute__((packed)) {
  int8_t gnssId;
  int8_t svid;
  int8_t reserved1;
  int8_t freqId;
  int8_t numWords;
  int8_t reserved2;
  int8_t version;
  int8_t reserved3;
} rxm_sfrbx_msg;

// Extra data count is in numWords
typedef struct __attribute__((packed)) {
  uint32_t dwrd;
} rxm_sfrbx_msg_extra;

namespace ublox {
  // protocol constants
  const uint8_t PREAMBLE1 = 0xb5;
  const uint8_t PREAMBLE2 = 0x62;

  // message classes
  const uint8_t CLASS_NAV = 0x01;
  const uint8_t CLASS_RXM = 0x02;

  // NAV messages
  const uint8_t MSG_NAV_PVT = 0x7;

  // RXM messages
  const uint8_t MSG_RXM_RAW = 0x15;
  const uint8_t MSG_RXM_SFRBX = 0x13;

  const int UBLOX_HEADER_SIZE = 6;
  const int UBLOX_CHECKSUM_SIZE = 2;
  const int UBLOX_MAX_MSG_SIZE = 65536;

  typedef std::map<uint8_t, std::vector<uint32_t>> subframes_map;

  class UbloxMsgParser {
    public:

      UbloxMsgParser();
      kj::Array<capnp::word> gen_solution();
      kj::Array<capnp::word> gen_raw();

      kj::Array<capnp::word> gen_nav_data();
      bool add_data(const uint8_t *incoming_data, uint32_t incoming_data_len, size_t &bytes_consumed);
      inline void reset() {bytes_in_parse_buf = 0;}
      inline uint8_t msg_class() {
        return msg_parse_buf[2];
      }

      inline uint8_t msg_id() {
        return msg_parse_buf[3];
      }
      inline int needed_bytes();

      void hexdump(uint8_t *d, int l) {
        for (int i = 0; i < l; i++) {
          if (i%0x10 == 0 && i != 0) printf("\n");
          printf("%02X ", d[i]);
        }
        printf("\n");
      }
    private:
      inline bool valid_cheksum();
      inline bool valid();
      inline bool valid_so_far();

      uint8_t msg_parse_buf[UBLOX_HEADER_SIZE + UBLOX_MAX_MSG_SIZE];
      int bytes_in_parse_buf;
      std::map<uint8_t, std::map<uint8_t, subframes_map>> nav_frame_buffer;
  };

}

typedef Message * (*poll_ubloxraw_msg_func)(Poller *poller);
typedef int (*send_gps_event_func)(PubSocket *s, const void *buf, size_t len);
int ubloxd_main(poll_ubloxraw_msg_func poll_func, send_gps_event_func send_func);
