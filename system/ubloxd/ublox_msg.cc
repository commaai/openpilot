#include "system/ubloxd/ublox_msg.h"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <utility>

#include "common/swaglog.h"

const double gpsPi = 3.1415926535898;
#define UBLOX_MSG_SIZE(hdr) (*(uint16_t *)&hdr[4])

inline static bool bit_to_bool(uint8_t val, int shifts) {
  return (bool)(val & (1 << shifts));
}

inline int UbloxMsgParser::needed_bytes() {
  // Msg header incomplete?
  if (bytes_in_parse_buf < ublox::UBLOX_HEADER_SIZE)
    return ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_CHECKSUM_SIZE - bytes_in_parse_buf;
  uint16_t needed = UBLOX_MSG_SIZE(msg_parse_buf) + ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_CHECKSUM_SIZE;
  // too much data
  if (needed < (uint16_t)bytes_in_parse_buf)
    return -1;
  return needed - (uint16_t)bytes_in_parse_buf;
}

inline bool UbloxMsgParser::valid_cheksum() {
  uint8_t ck_a = 0, ck_b = 0;
  for (int i = 2; i < bytes_in_parse_buf - ublox::UBLOX_CHECKSUM_SIZE; i++) {
    ck_a = (ck_a + msg_parse_buf[i]) & 0xFF;
    ck_b = (ck_b + ck_a) & 0xFF;
  }
  if (ck_a != msg_parse_buf[bytes_in_parse_buf - 2]) {
    LOGD("Checksum a mismatch: %02X, %02X", ck_a, msg_parse_buf[6]);
    return false;
  }
  if (ck_b != msg_parse_buf[bytes_in_parse_buf - 1]) {
    LOGD("Checksum b mismatch: %02X, %02X", ck_b, msg_parse_buf[7]);
    return false;
  }
  return true;
}

inline bool UbloxMsgParser::valid() {
  return bytes_in_parse_buf >= ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_CHECKSUM_SIZE &&
         needed_bytes() == 0 && valid_cheksum();
}

inline bool UbloxMsgParser::valid_so_far() {
  if (bytes_in_parse_buf > 0 && msg_parse_buf[0] != ublox::PREAMBLE1) {
    return false;
  }
  if (bytes_in_parse_buf > 1 && msg_parse_buf[1] != ublox::PREAMBLE2) {
    return false;
  }
  if (needed_bytes() == 0 && !valid()) {
    return false;
  }
  return true;
}

bool UbloxMsgParser::add_data(float log_time, const uint8_t *incoming_data, uint32_t incoming_data_len, size_t &bytes_consumed) {
  last_log_time = log_time;
  int needed = needed_bytes();
  if (needed > 0) {
    bytes_consumed = std::min((uint32_t)needed, incoming_data_len);
    // Add data to buffer
    memcpy(msg_parse_buf + bytes_in_parse_buf, incoming_data, bytes_consumed);
    bytes_in_parse_buf += bytes_consumed;
  } else {
    bytes_consumed = incoming_data_len;
  }

  // Validate msg format, detect invalid header and invalid checksum.
  while (!valid_so_far() && bytes_in_parse_buf != 0) {
    // Corrupted msg, drop a byte.
    bytes_in_parse_buf -= 1;
    if (bytes_in_parse_buf > 0)
      memmove(&msg_parse_buf[0], &msg_parse_buf[1], bytes_in_parse_buf);
  }

  // There is redundant data at the end of buffer, reset the buffer.
  if (needed_bytes() == -1) {
    bytes_in_parse_buf = 0;
  }
  return valid();
}


std::pair<std::string, kj::Array<capnp::word>> UbloxMsgParser::gen_msg() {
  std::string dat = data();
  kaitai::kstream stream(dat);

  ubx_t ubx_message(&stream);
  auto body = ubx_message.body();

  switch (ubx_message.msg_type()) {
  case 0x0107:
    return {"gpsLocationExternal", gen_nav_pvt(static_cast<ubx_t::nav_pvt_t*>(body))};
  case 0x0213: // UBX-RXM-SFRB (Broadcast Navigation Data Subframe)
    return {"ubloxGnss", gen_rxm_sfrbx(static_cast<ubx_t::rxm_sfrbx_t*>(body))};
  case 0x0215: // UBX-RXM-RAW (Multi-GNSS Raw Measurement Data)
    return {"ubloxGnss", gen_rxm_rawx(static_cast<ubx_t::rxm_rawx_t*>(body))};
  case 0x0a09:
    return {"ubloxGnss", gen_mon_hw(static_cast<ubx_t::mon_hw_t*>(body))};
  case 0x0a0b:
    return {"ubloxGnss", gen_mon_hw2(static_cast<ubx_t::mon_hw2_t*>(body))};
  case 0x0135:
    return {"ubloxGnss", gen_nav_sat(static_cast<ubx_t::nav_sat_t*>(body))};
  default:
    LOGE("Unknown message type %x", ubx_message.msg_type());
    return {"ubloxGnss", kj::Array<capnp::word>()};
  }
}


kj::Array<capnp::word> UbloxMsgParser::gen_nav_pvt(ubx_t::nav_pvt_t *msg) {
  MessageBuilder msg_builder;
  auto gpsLoc = msg_builder.initEvent().initGpsLocationExternal();
  gpsLoc.setSource(cereal::GpsLocationData::SensorSource::UBLOX);
  gpsLoc.setFlags(msg->flags());
  gpsLoc.setHasFix((msg->flags() % 2) == 1);
  gpsLoc.setLatitude(msg->lat() * 1e-07);
  gpsLoc.setLongitude(msg->lon() * 1e-07);
  gpsLoc.setAltitude(msg->height() * 1e-03);
  gpsLoc.setSpeed(msg->g_speed() * 1e-03);
  gpsLoc.setBearingDeg(msg->head_mot() * 1e-5);
  gpsLoc.setHorizontalAccuracy(msg->h_acc() * 1e-03);
  std::tm timeinfo = std::tm();
  timeinfo.tm_year = msg->year() - 1900;
  timeinfo.tm_mon = msg->month() - 1;
  timeinfo.tm_mday = msg->day();
  timeinfo.tm_hour = msg->hour();
  timeinfo.tm_min = msg->min();
  timeinfo.tm_sec = msg->sec();

  std::time_t utc_tt = timegm(&timeinfo);
  gpsLoc.setUnixTimestampMillis(utc_tt * 1e+03 + msg->nano() * 1e-06);
  float f[] = { msg->vel_n() * 1e-03f, msg->vel_e() * 1e-03f, msg->vel_d() * 1e-03f };
  gpsLoc.setVNED(f);
  gpsLoc.setVerticalAccuracy(msg->v_acc() * 1e-03);
  gpsLoc.setSpeedAccuracy(msg->s_acc() * 1e-03);
  gpsLoc.setBearingAccuracyDeg(msg->head_acc() * 1e-05);
  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::parse_gps_ephemeris(ubx_t::rxm_sfrbx_t *msg) {
  // GPS subframes are packed into 10x 4 bytes, each containing 3 actual bytes
  // We will first need to separate the data from the padding and parity
  auto body = *msg->body();
  assert(body.size() == 10);

  std::string subframe_data;
  subframe_data.reserve(30);
  for (uint32_t word : body) {
    word = word >> 6; // TODO: Verify parity
    subframe_data.push_back(word >> 16);
    subframe_data.push_back(word >> 8);
    subframe_data.push_back(word >> 0);
  }

  // Collect subframes in map and parse when we have all the parts
  {
    kaitai::kstream stream(subframe_data);
    gps_t subframe(&stream);

    int subframe_id = subframe.how()->subframe_id();
    if (subframe_id > 3 || subframe_id < 1) {
      // don't parse almanac subframes
      return kj::Array<capnp::word>();
    }
    gps_subframes[msg->sv_id()][subframe_id] = subframe_data;
  }

  // publish if subframes 1-3 have been collected
  if (gps_subframes[msg->sv_id()].size() == 3) {
    MessageBuilder msg_builder;
    auto eph = msg_builder.initEvent().initUbloxGnss().initEphemeris();
    eph.setSvId(msg->sv_id());

    int iode_s2 = 0;
    int iode_s3 = 0;
    int iodc_lsb = 0;
    int week;

    // Subframe 1
    {
      kaitai::kstream stream(gps_subframes[msg->sv_id()][1]);
      gps_t subframe(&stream);
      gps_t::subframe_1_t* subframe_1 = static_cast<gps_t::subframe_1_t*>(subframe.body());

      // Each message is incremented to be greater or equal than week 1877 (2015-12-27).
      //  To skip this use the current_time argument
      week = subframe_1->week_no();
      week += 1024;
      if (week < 1877) {
        week += 1024;
      }
      //eph.setGpsWeek(subframe_1->week_no());
      eph.setTgd(subframe_1->t_gd() * pow(2, -31));
      eph.setToc(subframe_1->t_oc() * pow(2, 4));
      eph.setAf2(subframe_1->af_2() * pow(2, -55));
      eph.setAf1(subframe_1->af_1() * pow(2, -43));
      eph.setAf0(subframe_1->af_0() * pow(2, -31));
      eph.setSvHealth(subframe_1->sv_health());
      eph.setTowCount(subframe.how()->tow_count());
      iodc_lsb = subframe_1->iodc_lsb();
    }

    // Subframe 2
    {
      kaitai::kstream stream(gps_subframes[msg->sv_id()][2]);
      gps_t subframe(&stream);
      gps_t::subframe_2_t* subframe_2 = static_cast<gps_t::subframe_2_t*>(subframe.body());

      // GPS week refers to current week, the ephemeris can be valid for the next
      // if toe equals 0, this can be verified by the TOW count if it is within the
      // last 2 hours of the week (gps ephemeris valid for 4hours)
      if (subframe_2->t_oe() == 0 and subframe.how()->tow_count()*6 >= (SECS_IN_WEEK - 2*SECS_IN_HR)){
        week += 1;
      }
      eph.setCrs(subframe_2->c_rs() * pow(2, -5));
      eph.setDeltaN(subframe_2->delta_n() * pow(2, -43) * gpsPi);
      eph.setM0(subframe_2->m_0() * pow(2, -31) * gpsPi);
      eph.setCuc(subframe_2->c_uc() * pow(2, -29));
      eph.setEcc(subframe_2->e() * pow(2, -33));
      eph.setCus(subframe_2->c_us() * pow(2, -29));
      eph.setA(pow(subframe_2->sqrt_a() * pow(2, -19), 2.0));
      eph.setToe(subframe_2->t_oe() * pow(2, 4));
      iode_s2 = subframe_2->iode();
    }

    // Subframe 3
    {
      kaitai::kstream stream(gps_subframes[msg->sv_id()][3]);
      gps_t subframe(&stream);
      gps_t::subframe_3_t* subframe_3 = static_cast<gps_t::subframe_3_t*>(subframe.body());

      eph.setCic(subframe_3->c_ic() * pow(2, -29));
      eph.setOmega0(subframe_3->omega_0() * pow(2, -31) * gpsPi);
      eph.setCis(subframe_3->c_is() * pow(2, -29));
      eph.setI0(subframe_3->i_0() * pow(2, -31) * gpsPi);
      eph.setCrc(subframe_3->c_rc() * pow(2, -5));
      eph.setOmega(subframe_3->omega() * pow(2, -31) * gpsPi);
      eph.setOmegaDot(subframe_3->omega_dot() * pow(2, -43) * gpsPi);
      eph.setIode(subframe_3->iode());
      eph.setIDot(subframe_3->idot() * pow(2, -43) * gpsPi);
      iode_s3 = subframe_3->iode();
    }

    eph.setToeWeek(week);
    eph.setTocWeek(week);

    gps_subframes[msg->sv_id()].clear();
    if (iodc_lsb != iode_s2 || iodc_lsb != iode_s3) {
      // data set cutover, reject ephemeris
      return kj::Array<capnp::word>();
    }
    return capnp::messageToFlatArray(msg_builder);
  }
  return kj::Array<capnp::word>();
}

kj::Array<capnp::word> UbloxMsgParser::parse_glonass_ephemeris(ubx_t::rxm_sfrbx_t *msg) {
  // This parser assumes that no 2 satellites of the same frequency
  // can be in view at the same time
  auto body = *msg->body();
  assert(body.size() == 4);
  {
    std::string string_data;
    string_data.reserve(16);
    for (uint32_t word : body) {
      for (int i = 3; i >= 0; i--)
        string_data.push_back(word >> 8*i);
    }

    kaitai::kstream stream(string_data);
    glonass_t gl_string(&stream);
    int string_number = gl_string.string_number();
    if (string_number < 1 || string_number > 5 || gl_string.idle_chip()) {
      // don't parse non immediate data, idle_chip == 0
      return kj::Array<capnp::word>();
    }

    // Check if new string either has same superframe_id or log transmission times make sense
    bool superframe_unknown = false;
    bool needs_clear = false;
    for (int i = 1; i <= 5; i++) {
      if (glonass_strings[msg->freq_id()].find(i) == glonass_strings[msg->freq_id()].end())
        continue;
      if (glonass_string_superframes[msg->freq_id()][i] == 0 || gl_string.superframe_number() == 0) {
        superframe_unknown = true;
      } else if (glonass_string_superframes[msg->freq_id()][i] != gl_string.superframe_number()) {
        needs_clear = true;
      }
      // Check if string times add up to being from the same frame
      // If superframe is known this is redundant
      // Strings are sent 2s apart and frames are 30s apart
      if (superframe_unknown &&
          std::abs((glonass_string_times[msg->freq_id()][i] - 2.0 * i) - (last_log_time - 2.0 * string_number)) > 10)
        needs_clear = true;
    }
    if (needs_clear) {
      glonass_strings[msg->freq_id()].clear();
      glonass_string_superframes[msg->freq_id()].clear();
      glonass_string_times[msg->freq_id()].clear();
    }
    glonass_strings[msg->freq_id()][string_number] = string_data;
    glonass_string_superframes[msg->freq_id()][string_number] = gl_string.superframe_number();
    glonass_string_times[msg->freq_id()][string_number] = last_log_time;
  }
  if (msg->sv_id() == 255) {
    // data can be decoded before identifying the SV number, in this case 255
    // is returned, which means "unknown"  (ublox p32)
    return kj::Array<capnp::word>();
  }

  // publish if strings 1-5 have been collected
  if (glonass_strings[msg->freq_id()].size() != 5) {
    return kj::Array<capnp::word>();
  }

  MessageBuilder msg_builder;
  auto eph = msg_builder.initEvent().initUbloxGnss().initGlonassEphemeris();
  eph.setSvId(msg->sv_id());
  eph.setFreqNum(msg->freq_id() - 7);

  uint16_t current_day = 0;
  uint16_t tk = 0;

  // string number 1
  {
    kaitai::kstream stream(glonass_strings[msg->freq_id()][1]);
    glonass_t gl_stream(&stream);
    glonass_t::string_1_t* data = static_cast<glonass_t::string_1_t*>(gl_stream.data());

    eph.setP1(data->p1());
    tk = data->t_k();
    eph.setTkDEPRECATED(tk);
    eph.setXVel(data->x_vel() * pow(2, -20));
    eph.setXAccel(data->x_accel() * pow(2, -30));
    eph.setX(data->x() * pow(2, -11));
  }

  // string number 2
  {
    kaitai::kstream stream(glonass_strings[msg->freq_id()][2]);
    glonass_t gl_stream(&stream);
    glonass_t::string_2_t* data = static_cast<glonass_t::string_2_t*>(gl_stream.data());

    eph.setSvHealth(data->b_n()>>2); // MSB indicates health
    eph.setP2(data->p2());
    eph.setTb(data->t_b());
    eph.setYVel(data->y_vel() * pow(2, -20));
    eph.setYAccel(data->y_accel() * pow(2, -30));
    eph.setY(data->y() * pow(2, -11));
  }

  // string number 3
  {
    kaitai::kstream stream(glonass_strings[msg->freq_id()][3]);
    glonass_t gl_stream(&stream);
    glonass_t::string_3_t* data = static_cast<glonass_t::string_3_t*>(gl_stream.data());

    eph.setP3(data->p3());
    eph.setGammaN(data->gamma_n() * pow(2, -40));
    eph.setSvHealth(eph.getSvHealth() | data->l_n());
    eph.setZVel(data->z_vel() * pow(2, -20));
    eph.setZAccel(data->z_accel() * pow(2, -30));
    eph.setZ(data->z() * pow(2, -11));
  }

  // string number 4
  {
    kaitai::kstream stream(glonass_strings[msg->freq_id()][4]);
    glonass_t gl_stream(&stream);
    glonass_t::string_4_t* data = static_cast<glonass_t::string_4_t*>(gl_stream.data());

    current_day = data->n_t();
    eph.setNt(current_day);
    eph.setTauN(data->tau_n() * pow(2, -30));
    eph.setDeltaTauN(data->delta_tau_n() * pow(2, -30));
    eph.setAge(data->e_n());
    eph.setP4(data->p4());
    eph.setSvURA(glonass_URA_lookup.at(data->f_t()));
    if (msg->sv_id() != data->n()) {
      LOGE("SV_ID != SLOT_NUMBER: %d %" PRIu64, msg->sv_id(), data->n());
    }
    eph.setSvType(data->m());
  }

  // string number 5
  {
    kaitai::kstream stream(glonass_strings[msg->freq_id()][5]);
    glonass_t gl_stream(&stream);
    glonass_t::string_5_t* data = static_cast<glonass_t::string_5_t*>(gl_stream.data());

    // string5 parsing is only needed to get the year, this can be removed and
    // the year can be fetched later in laika (note rollovers and leap year)
    eph.setN4(data->n_4());
    int tk_seconds = SECS_IN_HR * ((tk>>7) & 0x1F) + SECS_IN_MIN * ((tk>>1) & 0x3F) + (tk & 0x1) * 30;
    eph.setTkSeconds(tk_seconds);
  }

  glonass_strings[msg->freq_id()].clear();
  return capnp::messageToFlatArray(msg_builder);
}


kj::Array<capnp::word> UbloxMsgParser::gen_rxm_sfrbx(ubx_t::rxm_sfrbx_t *msg) {
  switch (msg->gnss_id()) {
    case ubx_t::gnss_type_t::GNSS_TYPE_GPS:
      return parse_gps_ephemeris(msg);
    case ubx_t::gnss_type_t::GNSS_TYPE_GLONASS:
      return parse_glonass_ephemeris(msg);
    default:
      return kj::Array<capnp::word>();
  }
}

kj::Array<capnp::word> UbloxMsgParser::gen_rxm_rawx(ubx_t::rxm_rawx_t *msg) {
  MessageBuilder msg_builder;
  auto mr = msg_builder.initEvent().initUbloxGnss().initMeasurementReport();
  mr.setRcvTow(msg->rcv_tow());
  mr.setGpsWeek(msg->week());
  mr.setLeapSeconds(msg->leap_s());
  mr.setGpsWeek(msg->week());

  auto mb = mr.initMeasurements(msg->num_meas());
  auto measurements = *msg->meas();
  for (int8_t i = 0; i < msg->num_meas(); i++) {
    mb[i].setSvId(measurements[i]->sv_id());
    mb[i].setPseudorange(measurements[i]->pr_mes());
    mb[i].setCarrierCycles(measurements[i]->cp_mes());
    mb[i].setDoppler(measurements[i]->do_mes());
    mb[i].setGnssId(measurements[i]->gnss_id());
    mb[i].setGlonassFrequencyIndex(measurements[i]->freq_id());
    mb[i].setLocktime(measurements[i]->lock_time());
    mb[i].setCno(measurements[i]->cno());
    mb[i].setPseudorangeStdev(0.01 * (pow(2, (measurements[i]->pr_stdev() & 15)))); // weird scaling, might be wrong
    mb[i].setCarrierPhaseStdev(0.004 * (measurements[i]->cp_stdev() & 15));
    mb[i].setDopplerStdev(0.002 * (pow(2, (measurements[i]->do_stdev() & 15)))); // weird scaling, might be wrong

    auto ts = mb[i].initTrackingStatus();
    auto trk_stat = measurements[i]->trk_stat();
    ts.setPseudorangeValid(bit_to_bool(trk_stat, 0));
    ts.setCarrierPhaseValid(bit_to_bool(trk_stat, 1));
    ts.setHalfCycleValid(bit_to_bool(trk_stat, 2));
    ts.setHalfCycleSubtracted(bit_to_bool(trk_stat, 3));
  }

  mr.setNumMeas(msg->num_meas());
  auto rs = mr.initReceiverStatus();
  rs.setLeapSecValid(bit_to_bool(msg->rec_stat(), 0));
  rs.setClkReset(bit_to_bool(msg->rec_stat(), 2));
  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::gen_nav_sat(ubx_t::nav_sat_t *msg) {
  MessageBuilder msg_builder;
  auto sr = msg_builder.initEvent().initUbloxGnss().initSatReport();
  sr.setITow(msg->itow());

  auto svs = sr.initSvs(msg->num_svs());
  auto svs_data = *msg->svs();
  for (int8_t i = 0; i < msg->num_svs(); i++) {
    svs[i].setSvId(svs_data[i]->sv_id());
    svs[i].setGnssId(svs_data[i]->gnss_id());
    svs[i].setFlagsBitfield(svs_data[i]->flags());
  }

  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::gen_mon_hw(ubx_t::mon_hw_t *msg) {
  MessageBuilder msg_builder;
  auto hwStatus = msg_builder.initEvent().initUbloxGnss().initHwStatus();
  hwStatus.setNoisePerMS(msg->noise_per_ms());
  hwStatus.setFlags(msg->flags());
  hwStatus.setAgcCnt(msg->agc_cnt());
  hwStatus.setAStatus((cereal::UbloxGnss::HwStatus::AntennaSupervisorState) msg->a_status());
  hwStatus.setAPower((cereal::UbloxGnss::HwStatus::AntennaPowerStatus) msg->a_power());
  hwStatus.setJamInd(msg->jam_ind());
  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::gen_mon_hw2(ubx_t::mon_hw2_t *msg) {
  MessageBuilder msg_builder;
  auto hwStatus = msg_builder.initEvent().initUbloxGnss().initHwStatus2();
  hwStatus.setOfsI(msg->ofs_i());
  hwStatus.setMagI(msg->mag_i());
  hwStatus.setOfsQ(msg->ofs_q());
  hwStatus.setMagQ(msg->mag_q());

  switch (msg->cfg_source()) {
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_ROM:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::ROM);
      break;
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_OTP:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::OTP);
      break;
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_CONFIG_PINS:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::CONFIGPINS);
      break;
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_FLASH:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::FLASH);
      break;
    default:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::UNDEFINED);
      break;
  }

  hwStatus.setLowLevCfg(msg->low_lev_cfg());
  hwStatus.setPostStatus(msg->post_status());

  return capnp::messageToFlatArray(msg_builder);
}
