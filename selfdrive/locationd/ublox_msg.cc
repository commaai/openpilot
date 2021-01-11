#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <ctime>
#include <chrono>

#include "common/swaglog.h"

#include "ublox_msg.h"

#define UBLOX_MSG_SIZE(hdr) (*(uint16_t *)&hdr[4])
#define GET_FIELD_U(w, nb, pos) (((w) >> (pos)) & ((1<<(nb))-1))

namespace ublox {

inline int twos_complement(uint32_t v, uint32_t nb) {
  int sign = v >> (nb - 1);
  int value = v;
  if(sign != 0)
    value = value - (1 << nb);
  return value;
}

inline int GET_FIELD_S(uint32_t w, uint32_t nb, uint32_t pos) {
  int v = GET_FIELD_U(w, nb, pos);
  return twos_complement(v, nb);
}

class EphemerisData {
  public:
    EphemerisData(uint8_t svId, subframes_map &subframes) {
      this->svId = svId;
      int week_no = GET_FIELD_U(subframes[1][2+0], 10, 20);
      int t_gd = GET_FIELD_S(subframes[1][2+4], 8, 6);
      int iodc = (GET_FIELD_U(subframes[1][2+0], 2, 6) << 8) | GET_FIELD_U(
        subframes[1][2+5], 8, 22);

      int t_oc = GET_FIELD_U(subframes[1][2+5], 16, 6);
      int a_f2 = GET_FIELD_S(subframes[1][2+6], 8, 22);
      int a_f1 = GET_FIELD_S(subframes[1][2+6], 16, 6);
      int a_f0 = GET_FIELD_S(subframes[1][2+7], 22, 8);

      int c_rs = GET_FIELD_S(subframes[2][2+0], 16, 6);
      int delta_n = GET_FIELD_S(subframes[2][2+1], 16, 14);
      int m_0 = (GET_FIELD_S(subframes[2][2+1], 8, 6) << 24) | GET_FIELD_U(
        subframes[2][2+2], 24, 6);
      int c_uc = GET_FIELD_S(subframes[2][2+3], 16, 14);
      int e = (GET_FIELD_U(subframes[2][2+3], 8, 6) << 24) | GET_FIELD_U(subframes[2][2+4], 24, 6);
      int c_us = GET_FIELD_S(subframes[2][2+5], 16, 14);
      uint32_t a_powhalf = (GET_FIELD_U(subframes[2][2+5], 8, 6) << 24) | GET_FIELD_U(
        subframes[2][2+6], 24, 6);
      int t_oe = GET_FIELD_U(subframes[2][2+7], 16, 14);

      int c_ic = GET_FIELD_S(subframes[3][2+0], 16, 14);
      int omega_0 = (GET_FIELD_S(subframes[3][2+0], 8, 6) << 24) | GET_FIELD_U(
        subframes[3][2+1], 24, 6);
      int c_is = GET_FIELD_S(subframes[3][2+2], 16, 14);
      int i_0 = (GET_FIELD_S(subframes[3][2+2], 8, 6) << 24) | GET_FIELD_U(
        subframes[3][2+3], 24, 6);
      int c_rc = GET_FIELD_S(subframes[3][2+4], 16, 14);
      int w = (GET_FIELD_S(subframes[3][2+4], 8, 6) << 24) | GET_FIELD_U(subframes[3][5], 24, 6);
      int omega_dot = GET_FIELD_S(subframes[3][2+6], 24, 6);
      int idot = GET_FIELD_S(subframes[3][2+7], 14, 8);

      this->_rsvd1 = GET_FIELD_U(subframes[1][2+1], 23, 6);
      this->_rsvd2 = GET_FIELD_U(subframes[1][2+2], 24, 6);
      this->_rsvd3 = GET_FIELD_U(subframes[1][2+3], 24, 6);
      this->_rsvd4 = GET_FIELD_U(subframes[1][2+4], 16, 14);
      this->aodo = GET_FIELD_U(subframes[2][2+7], 5, 8);

      double gpsPi = 3.1415926535898;

      // now form variables in radians, meters and seconds etc
      this->Tgd = t_gd * pow(2, -31);
      this->A = pow(a_powhalf * pow(2, -19), 2.0);
      this->cic = c_ic * pow(2, -29);
      this->cis = c_is * pow(2, -29);
      this->crc = c_rc * pow(2, -5);
      this->crs = c_rs * pow(2, -5);
      this->cuc = c_uc * pow(2, -29);
      this->cus = c_us * pow(2, -29);
      this->deltaN = delta_n * pow(2, -43) * gpsPi;
      this->ecc = e * pow(2, -33);
      this->i0 = i_0 * pow(2, -31) * gpsPi;
      this->idot = idot * pow(2, -43) * gpsPi;
      this->M0 = m_0 * pow(2, -31) * gpsPi;
      this->omega = w * pow(2, -31) * gpsPi;
      this->omega_dot = omega_dot * pow(2, -43) * gpsPi;
      this->omega0 = omega_0 * pow(2, -31) * gpsPi;
      this->toe = t_oe * pow(2, 4);

      this->toc = t_oc * pow(2, 4);
      this->gpsWeek = week_no;
      this->af0 = a_f0 * pow(2, -31);
      this->af1 = a_f1 * pow(2, -43);
      this->af2 = a_f2 * pow(2, -55);

      uint32_t iode1 = GET_FIELD_U(subframes[2][2+0], 8, 22);
      uint32_t iode2 = GET_FIELD_U(subframes[3][2+7], 8, 22);
      this->valid = (iode1 == iode2) && (iode1 == (iodc & 0xff));
      this->iode = iode1;

      if (GET_FIELD_U(subframes[4][2+0], 6, 22) == 56 &&
        GET_FIELD_U(subframes[4][2+0], 2, 28) == 1 &&
        GET_FIELD_U(subframes[5][2+0], 2, 28) == 1) {
        double a0 = GET_FIELD_S(subframes[4][2], 8, 14) * pow(2, -30);
        double a1 = GET_FIELD_S(subframes[4][2], 8, 6) * pow(2, -27);
        double a2 = GET_FIELD_S(subframes[4][3], 8, 22) * pow(2, -24);
        double a3 = GET_FIELD_S(subframes[4][3], 8, 14) * pow(2, -24);
        double b0 = GET_FIELD_S(subframes[4][3], 8, 6) * pow(2, 11);
        double b1 = GET_FIELD_S(subframes[4][4], 8, 22) * pow(2, 14);
        double b2 = GET_FIELD_S(subframes[4][4], 8, 14) * pow(2, 16);
        double b3 = GET_FIELD_S(subframes[4][4], 8, 6) * pow(2, 16);
        this->ionoAlpha[0] = a0;this->ionoAlpha[1] = a1;this->ionoAlpha[2] = a2;this->ionoAlpha[3] = a3;
        this->ionoBeta[0] = b0;this->ionoBeta[1] = b1;this->ionoBeta[2] = b2;this->ionoBeta[3] = b3;
        this->ionoCoeffsValid = true;
      } else {
        this->ionoCoeffsValid = false;
      }
    }
    uint16_t svId;
    double Tgd, A, cic, cis, crc, crs, cuc, cus, deltaN, ecc, i0, idot, M0, omega, omega_dot, omega0, toe, toc;
    uint32_t gpsWeek, iode, _rsvd1, _rsvd2, _rsvd3, _rsvd4, aodo;
    double af0, af1, af2;
    bool valid;
    double ionoAlpha[4], ionoBeta[4];
    bool ionoCoeffsValid;
};

UbloxMsgParser::UbloxMsgParser() :bytes_in_parse_buf(0) {
  nav_frame_buffer[0U] = std::map<uint8_t, subframes_map>();
  for(int i = 1;i < 33;i++)
    nav_frame_buffer[0U][i] = subframes_map();
}

inline int UbloxMsgParser::needed_bytes() {
  // Msg header incomplete?
  if(bytes_in_parse_buf < UBLOX_HEADER_SIZE)
    return UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE - bytes_in_parse_buf;
  uint16_t needed = UBLOX_MSG_SIZE(msg_parse_buf) + UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE;
  // too much data
  if(needed < (uint16_t)bytes_in_parse_buf)
    return -1;
  return needed - (uint16_t)bytes_in_parse_buf;
}

inline bool UbloxMsgParser::valid_cheksum() {
  uint8_t ck_a = 0, ck_b = 0;
  for(int i = 2; i < bytes_in_parse_buf - UBLOX_CHECKSUM_SIZE;i++) {
    ck_a = (ck_a + msg_parse_buf[i]) & 0xFF;
    ck_b = (ck_b + ck_a) & 0xFF;
  }
  if(ck_a != msg_parse_buf[bytes_in_parse_buf - 2]) {
    LOGD("Checksum a mismtach: %02X, %02X", ck_a, msg_parse_buf[6]);
    return false;
  }
  if(ck_b != msg_parse_buf[bytes_in_parse_buf - 1]) {
    LOGD("Checksum b mismtach: %02X, %02X", ck_b, msg_parse_buf[7]);
    return false;
  }
  return true;
}

inline bool UbloxMsgParser::valid() {
  return bytes_in_parse_buf >= UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE &&
         needed_bytes() == 0 && valid_cheksum();
}

inline bool UbloxMsgParser::valid_so_far() {
  if(bytes_in_parse_buf > 0 && msg_parse_buf[0] != PREAMBLE1) {
    //LOGD("PREAMBLE1 invalid, %02X.", msg_parse_buf[0]);
    return false;
  }
  if(bytes_in_parse_buf > 1 && msg_parse_buf[1] != PREAMBLE2) {
    //LOGD("PREAMBLE2 invalid, %02X.", msg_parse_buf[1]);
    return false;
  }
  if(needed_bytes() == 0 && !valid()) {
    return false;
  }
  return true;
}

kj::Array<capnp::word> UbloxMsgParser::gen_solution() {
  nav_pvt_msg *msg = (nav_pvt_msg *)&msg_parse_buf[UBLOX_HEADER_SIZE];
  MessageBuilder msg_builder;
  auto gpsLoc = msg_builder.initEvent().initGpsLocationExternal();
  gpsLoc.setSource(cereal::GpsLocationData::SensorSource::UBLOX);
  gpsLoc.setFlags(msg->flags);
  gpsLoc.setLatitude(msg->lat * 1e-07);
  gpsLoc.setLongitude(msg->lon * 1e-07);
  gpsLoc.setAltitude(msg->height * 1e-03);
  gpsLoc.setSpeed(msg->gSpeed * 1e-03);
  gpsLoc.setBearing(msg->headMot * 1e-5);
  gpsLoc.setAccuracy(msg->hAcc * 1e-03);
  std::tm timeinfo = std::tm();
  timeinfo.tm_year = msg->year - 1900;
  timeinfo.tm_mon = msg->month - 1;
  timeinfo.tm_mday = msg->day;
  timeinfo.tm_hour = msg->hour;
  timeinfo.tm_min = msg->min;
  timeinfo.tm_sec = msg->sec;
  std::time_t utc_tt = timegm(&timeinfo);
  gpsLoc.setTimestamp(utc_tt * 1e+03 + msg->nano * 1e-06);
  float f[] = { msg->velN * 1e-03f, msg->velE * 1e-03f, msg->velD * 1e-03f };
  gpsLoc.setVNED(f);
  gpsLoc.setVerticalAccuracy(msg->vAcc * 1e-03);
  gpsLoc.setSpeedAccuracy(msg->sAcc * 1e-03);
  gpsLoc.setBearingAccuracy(msg->headAcc * 1e-05);
  return capnp::messageToFlatArray(msg_builder);
}

inline bool bit_to_bool(uint8_t val, int shifts) {
  return (bool)(val & (1 << shifts));
}

kj::Array<capnp::word> UbloxMsgParser::gen_raw() {
  rxm_raw_msg *msg = (rxm_raw_msg *)&msg_parse_buf[UBLOX_HEADER_SIZE];
  if(bytes_in_parse_buf != (
    UBLOX_HEADER_SIZE + sizeof(rxm_raw_msg) + msg->numMeas * sizeof(rxm_raw_msg_extra) + UBLOX_CHECKSUM_SIZE
    )) {
    LOGD("Invalid measurement size %u, %u, %u, %u", msg->numMeas, bytes_in_parse_buf, sizeof(rxm_raw_msg_extra), sizeof(rxm_raw_msg));
    return kj::Array<capnp::word>();
  }
  rxm_raw_msg_extra *measurements = (rxm_raw_msg_extra *)&msg_parse_buf[UBLOX_HEADER_SIZE + sizeof(rxm_raw_msg)];
  MessageBuilder msg_builder;
  auto mr = msg_builder.initEvent().initUbloxGnss().initMeasurementReport();
  mr.setRcvTow(msg->rcvTow);
  mr.setGpsWeek(msg->week);
  mr.setLeapSeconds(msg->leapS);
  mr.setGpsWeek(msg->week);
  auto mb = mr.initMeasurements(msg->numMeas);
  for(int8_t i = 0; i < msg->numMeas; i++) {
    mb[i].setSvId(measurements[i].svId);
    mb[i].setSigId(measurements[i].sigId);
    mb[i].setPseudorange(measurements[i].prMes);
    mb[i].setCarrierCycles(measurements[i].cpMes);
    mb[i].setDoppler(measurements[i].doMes);
    mb[i].setGnssId(measurements[i].gnssId);
    mb[i].setGlonassFrequencyIndex(measurements[i].freqId);
    mb[i].setLocktime(measurements[i].locktime);
    mb[i].setCno(measurements[i].cno);
    mb[i].setPseudorangeStdev(0.01*(pow(2, (measurements[i].prStdev & 15)))); // weird scaling, might be wrong
    mb[i].setCarrierPhaseStdev(0.004*(measurements[i].cpStdev & 15));
    mb[i].setDopplerStdev(0.002*(pow(2, (measurements[i].doStdev & 15)))); // weird scaling, might be wrong
    auto ts = mb[i].initTrackingStatus();
    ts.setPseudorangeValid(bit_to_bool(measurements[i].trkStat, 0));
    ts.setCarrierPhaseValid(bit_to_bool(measurements[i].trkStat, 1));
    ts.setHalfCycleValid(bit_to_bool(measurements[i].trkStat, 2));
    ts.setHalfCycleSubtracted(bit_to_bool(measurements[i].trkStat, 3));
  }

  mr.setNumMeas(msg->numMeas);
  auto rs = mr.initReceiverStatus();
  rs.setLeapSecValid(bit_to_bool(msg->recStat, 0));
  rs.setClkReset(bit_to_bool(msg->recStat, 2));
  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::gen_nav_data() {
  rxm_sfrbx_msg *msg = (rxm_sfrbx_msg *)&msg_parse_buf[UBLOX_HEADER_SIZE];
  if(bytes_in_parse_buf != (
    UBLOX_HEADER_SIZE + sizeof(rxm_sfrbx_msg) + msg->numWords * sizeof(rxm_sfrbx_msg_extra) + UBLOX_CHECKSUM_SIZE
    )) {
    LOGD("Invalid sfrbx words size %u, %u, %u, %u", msg->numWords, bytes_in_parse_buf, sizeof(rxm_raw_msg_extra), sizeof(rxm_raw_msg));
    return kj::Array<capnp::word>();
  }
  rxm_sfrbx_msg_extra *measurements = (rxm_sfrbx_msg_extra *)&msg_parse_buf[UBLOX_HEADER_SIZE + sizeof(rxm_sfrbx_msg)];
  if(msg->gnssId  == 0) {
    uint8_t subframeId =  GET_FIELD_U(measurements[1].dwrd, 3, 8);
    std::vector<uint32_t> words;
    for(int i = 0; i < msg->numWords;i++)
      words.push_back(measurements[i].dwrd);

    subframes_map &map = nav_frame_buffer[msg->gnssId][msg->svid];
    if (subframeId == 1) {
      map = subframes_map();
      map[subframeId] = words;
    } else if (map.find(subframeId-1) != map.end()) {
      map[subframeId] = words;
    }
    if(map.size() == 5) {
      EphemerisData ephem_data(msg->svid, map);
      MessageBuilder msg_builder;
      auto eph = msg_builder.initEvent().initUbloxGnss().initEphemeris();
      eph.setSvId(ephem_data.svId);
      eph.setToc(ephem_data.toc);
      eph.setGpsWeek(ephem_data.gpsWeek);
      eph.setAf0(ephem_data.af0);
      eph.setAf1(ephem_data.af1);
      eph.setAf2(ephem_data.af2);
      eph.setIode(ephem_data.iode);
      eph.setCrs(ephem_data.crs);
      eph.setDeltaN(ephem_data.deltaN);
      eph.setM0(ephem_data.M0);
      eph.setCuc(ephem_data.cuc);
      eph.setEcc(ephem_data.ecc);
      eph.setCus(ephem_data.cus);
      eph.setA(ephem_data.A);
      eph.setToe(ephem_data.toe);
      eph.setCic(ephem_data.cic);
      eph.setOmega0(ephem_data.omega0);
      eph.setCis(ephem_data.cis);
      eph.setI0(ephem_data.i0);
      eph.setCrc(ephem_data.crc);
      eph.setOmega(ephem_data.omega);
      eph.setOmegaDot(ephem_data.omega_dot);
      eph.setIDot(ephem_data.idot);
      eph.setTgd(ephem_data.Tgd);
      eph.setIonoCoeffsValid(ephem_data.ionoCoeffsValid);
      if(ephem_data.ionoCoeffsValid) {
        eph.setIonoAlpha(ephem_data.ionoAlpha);
        eph.setIonoBeta(ephem_data.ionoBeta);
      } else {
        eph.setIonoAlpha(kj::ArrayPtr<const double>());
        eph.setIonoBeta(kj::ArrayPtr<const double>());
      }
      return capnp::messageToFlatArray(msg_builder);
    }
  }
  return kj::Array<capnp::word>();
}

kj::Array<capnp::word> UbloxMsgParser::gen_mon_hw() {
  mon_hw_msg *msg = (mon_hw_msg *)&msg_parse_buf[UBLOX_HEADER_SIZE];

  MessageBuilder msg_builder;
  auto hwStatus = msg_builder.initEvent().initUbloxGnss().initHwStatus();
  hwStatus.setNoisePerMS(msg->noisePerMS);
  hwStatus.setAgcCnt(msg->agcCnt);
  hwStatus.setAStatus((cereal::UbloxGnss::HwStatus::AntennaSupervisorState) msg->aStatus);
  hwStatus.setAPower((cereal::UbloxGnss::HwStatus::AntennaPowerStatus) msg->aPower);
  hwStatus.setJamInd(msg->jamInd);
  return capnp::messageToFlatArray(msg_builder);
}

bool UbloxMsgParser::add_data(const uint8_t *incoming_data, uint32_t incoming_data_len, size_t &bytes_consumed) {
  int needed = needed_bytes();
  if(needed > 0) {
    bytes_consumed = std::min((uint32_t)needed, incoming_data_len );
    // Add data to buffer
    memcpy(msg_parse_buf + bytes_in_parse_buf, incoming_data, bytes_consumed);
    bytes_in_parse_buf += bytes_consumed;
  } else {
    bytes_consumed = incoming_data_len;
  }
  // Validate msg format, detect invalid header and invalid checksum.
  while(!valid_so_far() && bytes_in_parse_buf != 0) {
    //LOGD("Drop corrupt data, remained in buf: %u", bytes_in_parse_buf);
    // Corrupted msg, drop a byte.
    bytes_in_parse_buf -= 1;
    if(bytes_in_parse_buf > 0)
      memmove(&msg_parse_buf[0], &msg_parse_buf[1], bytes_in_parse_buf);
  }
  // There is redundant data at the end of buffer, reset the buffer.
  if(needed_bytes() == -1)
    bytes_in_parse_buf = 0;
  return valid();
}

}
