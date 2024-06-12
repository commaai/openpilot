#include <iostream>
#include <vector>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <ctime>

#include "catch2/catch.hpp"
#include "system/ubloxd/generated/glonass.h"

typedef std::vector<std::pair<int, int64_t>> string_data;

#define IDLE_CHIP_IDX 0
#define STRING_NUMBER_IDX 1
// string data 1-5
#define HC_IDX 0
#define PAD1_IDX 1
#define SUPERFRAME_IDX 2
#define PAD2_IDX 3
#define FRAME_IDX 4

// Indexes for string number 1
#define ST1_NU_IDX 2
#define ST1_P1_IDX 3
#define ST1_T_K_IDX 4
#define ST1_X_VEL_S_IDX 5
#define ST1_X_VEL_V_IDX 6
#define ST1_X_ACCEL_S_IDX 7
#define ST1_X_ACCEL_V_IDX 8
#define ST1_X_S_IDX 9
#define ST1_X_V_IDX 10
#define ST1_HC_OFF 11

// Indexes for string number 2
#define ST2_BN_IDX 2
#define ST2_P2_IDX 3
#define ST2_TB_IDX 4
#define ST2_NU_IDX 5
#define ST2_Y_VEL_S_IDX 6
#define ST2_Y_VEL_V_IDX 7
#define ST2_Y_ACCEL_S_IDX 8
#define ST2_Y_ACCEL_V_IDX 9
#define ST2_Y_S_IDX 10
#define ST2_Y_V_IDX 11
#define ST2_HC_OFF 12

// Indexes for string number 3
#define ST3_P3_IDX 2
#define ST3_GAMMA_N_S_IDX 3
#define ST3_GAMMA_N_V_IDX 4
#define ST3_NU_1_IDX 5
#define ST3_P_IDX 6
#define ST3_L_N_IDX 7
#define ST3_Z_VEL_S_IDX 8
#define ST3_Z_VEL_V_IDX 9
#define ST3_Z_ACCEL_S_IDX 10
#define ST3_Z_ACCEL_V_IDX 11
#define ST3_Z_S_IDX 12
#define ST3_Z_V_IDX 13
#define ST3_HC_OFF 14

// Indexes for string number 4
#define ST4_TAU_N_S_IDX 2
#define ST4_TAU_N_V_IDX 3
#define ST4_DELTA_TAU_N_S_IDX 4
#define ST4_DELTA_TAU_N_V_IDX 5
#define ST4_E_N_IDX 6
#define ST4_NU_1_IDX 7
#define ST4_P4_IDX 8
#define ST4_F_T_IDX 9
#define ST4_NU_2_IDX 10
#define ST4_N_T_IDX 11
#define ST4_N_IDX 12
#define ST4_M_IDX 13
#define ST4_HC_OFF 14

// Indexes for string number 5
#define ST5_N_A_IDX 2
#define ST5_TAU_C_IDX 3
#define ST5_NU_IDX 4
#define ST5_N_4_IDX 5
#define ST5_TAU_GPS_IDX 6
#define ST5_L_N_IDX 7
#define ST5_HC_OFF 8

// Indexes for non immediate
#define ST6_DATA_1_IDX 2
#define ST6_DATA_2_IDX 3
#define ST6_HC_OFF 4


std::string generate_inp_data(string_data& data) {
  std::string inp_data = "";
  for (auto& [b, v] : data) {
    std::string tmp = std::bitset<64>(v).to_string();
    inp_data += tmp.substr(64-b, b);
  }
  assert(inp_data.size() == 128);

  std::string string_data;
  string_data.reserve(16);
  for (int i = 0; i < 128; i+=8) {
    std::string substr = inp_data.substr(i, 8);
    string_data.push_back((uint8_t)std::stoi(substr.c_str(), 0, 2));
  }

  return string_data;
}

string_data generate_string_data(uint8_t string_number) {

  srand((unsigned)time(0));
  string_data data; //<bit length, value>
  data.push_back({1, 0}); // idle chip
  data.push_back({4, string_number}); // string number

  if (string_number == 1) {
    data.push_back({2, 3}); // not_used
    data.push_back({2, 1}); // p1
    data.push_back({12, 113}); // t_k
    data.push_back({1, rand() & 1}); // x_vel_sign
    data.push_back({23, 7122}); // x_vel_value
    data.push_back({1, rand() & 1}); // x_accel_sign
    data.push_back({4, 3}); // x_accel_value
    data.push_back({1, rand() & 1}); // x_sign
    data.push_back({26, 33554431}); // x_value
  } else if (string_number == 2) {
    data.push_back({3, 3}); // b_n
    data.push_back({1, 1}); // p2
    data.push_back({7, 123}); // t_b
    data.push_back({5, 31}); // not_used
    data.push_back({1, rand() & 1}); // y_vel_sign
    data.push_back({23, 7422}); // y_vel_value
    data.push_back({1, rand() & 1}); // y_accel_sign
    data.push_back({4, 3}); // y_accel_value
    data.push_back({1, rand() & 1}); // y_sign
    data.push_back({26, 67108863}); // y_value
  } else if (string_number == 3) {
    data.push_back({1, 0}); // p3
    data.push_back({1, 1}); // gamma_n_sign
    data.push_back({10, 123}); // gamma_n_value
    data.push_back({1, 0}); // not_used
    data.push_back({2, 2}); // p
    data.push_back({1, 1}); // l_n
    data.push_back({1, rand() & 1}); // z_vel_sign
    data.push_back({23, 1337}); // z_vel_value
    data.push_back({1, rand() & 1}); // z_accel_sign
    data.push_back({4, 9}); // z_accel_value
    data.push_back({1, rand() & 1}); // z_sign
    data.push_back({26, 100023}); // z_value
  } else if (string_number == 4) {
    data.push_back({1, rand() & 1}); // tau_n_sign
    data.push_back({21, 197152}); // tau_n_value
    data.push_back({1, rand() & 1}); // delta_tau_n_sign
    data.push_back({4, 4}); // delta_tau_n_value
    data.push_back({5, 0}); // e_n
    data.push_back({14, 2}); // not_used_1
    data.push_back({1, 1}); // p4
    data.push_back({4, 9}); // f_t
    data.push_back({3, 3}); // not_used_2
    data.push_back({11, 2047}); // n_t
    data.push_back({5, 2}); // n
    data.push_back({2, 1}); // m
  } else if (string_number == 5) {
    data.push_back({11, 2047}); // n_a
    data.push_back({32, 4294767295}); // tau_c
    data.push_back({1, 0}); // not_used_1
    data.push_back({5, 2}); // n_4
    data.push_back({22, 4114304}); // tau_gps
    data.push_back({1, 0}); // l_n
  } else { // non-immediate data is not parsed
    data.push_back({64, rand()}); // data_1
    data.push_back({8, 6}); // data_2
  }

  data.push_back({8, rand() & 0xFF}); // hamming code
  data.push_back({11, rand() & 0x7FF}); // pad
  data.push_back({16, rand() & 0xFFFF}); // superframe
  data.push_back({8, rand() & 0xFF}); // pad
  data.push_back({8, rand() & 0xFF}); // frame
  return data;
}

TEST_CASE("parse_string_number_1"){
  string_data data = generate_string_data(1);
  std::string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);
  REQUIRE(gl_string.hamming_code() == data[ST1_HC_OFF + HC_IDX].second);
  REQUIRE(gl_string.pad_1() == data[ST1_HC_OFF + PAD1_IDX].second);
  REQUIRE(gl_string.superframe_number() == data[ST1_HC_OFF + SUPERFRAME_IDX].second);
  REQUIRE(gl_string.pad_2() == data[ST1_HC_OFF + PAD2_IDX].second);
  REQUIRE(gl_string.frame_number() == data[ST1_HC_OFF + FRAME_IDX].second);

  kaitai::kstream str1(inp_data);
  glonass_t str1_data(&str1);
  glonass_t::string_1_t* s1 = static_cast<glonass_t::string_1_t*>(str1_data.data());

  REQUIRE(s1->not_used() == data[ST1_NU_IDX].second);
  REQUIRE(s1->p1() == data[ST1_P1_IDX].second);
  REQUIRE(s1->t_k() == data[ST1_T_K_IDX].second);

  int mul = s1->x_vel_sign() ? (-1) : 1;
  REQUIRE(s1->x_vel() == (data[ST1_X_VEL_V_IDX].second * mul));
  mul = s1->x_accel_sign() ? (-1) : 1;
  REQUIRE(s1->x_accel() == (data[ST1_X_ACCEL_V_IDX].second * mul));
  mul = s1->x_sign() ? (-1) : 1;
  REQUIRE(s1->x() == (data[ST1_X_V_IDX].second * mul));
}

TEST_CASE("parse_string_number_2"){
  string_data data = generate_string_data(2);
  std::string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);
  REQUIRE(gl_string.hamming_code() == data[ST2_HC_OFF + HC_IDX].second);
  REQUIRE(gl_string.pad_1() == data[ST2_HC_OFF + PAD1_IDX].second);
  REQUIRE(gl_string.superframe_number() == data[ST2_HC_OFF + SUPERFRAME_IDX].second);
  REQUIRE(gl_string.pad_2() == data[ST2_HC_OFF + PAD2_IDX].second);
  REQUIRE(gl_string.frame_number() == data[ST2_HC_OFF + FRAME_IDX].second);

  kaitai::kstream str2(inp_data);
  glonass_t str2_data(&str2);
  glonass_t::string_2_t* s2 = static_cast<glonass_t::string_2_t*>(str2_data.data());

  REQUIRE(s2->b_n() == data[ST2_BN_IDX].second);
  REQUIRE(s2->not_used() == data[ST2_NU_IDX].second);
  REQUIRE(s2->p2() == data[ST2_P2_IDX].second);
  REQUIRE(s2->t_b() == data[ST2_TB_IDX].second);
  int mul = s2->y_vel_sign() ? (-1) : 1;
  REQUIRE(s2->y_vel() == (data[ST2_Y_VEL_V_IDX].second * mul));
  mul = s2->y_accel_sign() ? (-1) : 1;
  REQUIRE(s2->y_accel() == (data[ST2_Y_ACCEL_V_IDX].second * mul));
  mul = s2->y_sign() ? (-1) : 1;
  REQUIRE(s2->y() == (data[ST2_Y_V_IDX].second * mul));
}

TEST_CASE("parse_string_number_3"){
  string_data data = generate_string_data(3);
  std::string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);
  REQUIRE(gl_string.hamming_code() == data[ST3_HC_OFF + HC_IDX].second);
  REQUIRE(gl_string.pad_1() == data[ST3_HC_OFF + PAD1_IDX].second);
  REQUIRE(gl_string.superframe_number() == data[ST3_HC_OFF + SUPERFRAME_IDX].second);
  REQUIRE(gl_string.pad_2() == data[ST3_HC_OFF + PAD2_IDX].second);
  REQUIRE(gl_string.frame_number() == data[ST3_HC_OFF + FRAME_IDX].second);

  kaitai::kstream str3(inp_data);
  glonass_t str3_data(&str3);
  glonass_t::string_3_t* s3 = static_cast<glonass_t::string_3_t*>(str3_data.data());

  REQUIRE(s3->p3() == data[ST3_P3_IDX].second);
  int mul = s3->gamma_n_sign() ? (-1) : 1;
  REQUIRE(s3->gamma_n() == (data[ST3_GAMMA_N_V_IDX].second * mul));
  REQUIRE(s3->not_used() == data[ST3_NU_1_IDX].second);
  REQUIRE(s3->p() == data[ST3_P_IDX].second);
  REQUIRE(s3->l_n() == data[ST3_L_N_IDX].second);
  mul = s3->z_vel_sign() ? (-1) : 1;
  REQUIRE(s3->z_vel() == (data[ST3_Z_VEL_V_IDX].second * mul));
  mul = s3->z_accel_sign() ? (-1) : 1;
  REQUIRE(s3->z_accel() == (data[ST3_Z_ACCEL_V_IDX].second * mul));
  mul = s3->z_sign() ? (-1) : 1;
  REQUIRE(s3->z() == (data[ST3_Z_V_IDX].second * mul));
}

TEST_CASE("parse_string_number_4"){
  string_data data = generate_string_data(4);
  std::string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);
  REQUIRE(gl_string.hamming_code() == data[ST4_HC_OFF + HC_IDX].second);
  REQUIRE(gl_string.pad_1() == data[ST4_HC_OFF + PAD1_IDX].second);
  REQUIRE(gl_string.superframe_number() == data[ST4_HC_OFF + SUPERFRAME_IDX].second);
  REQUIRE(gl_string.pad_2() == data[ST4_HC_OFF + PAD2_IDX].second);
  REQUIRE(gl_string.frame_number() == data[ST4_HC_OFF + FRAME_IDX].second);

  kaitai::kstream str4(inp_data);
  glonass_t str4_data(&str4);
  glonass_t::string_4_t* s4 = static_cast<glonass_t::string_4_t*>(str4_data.data());

  int mul = s4->tau_n_sign() ? (-1) : 1;
  REQUIRE(s4->tau_n() == (data[ST4_TAU_N_V_IDX].second * mul));
  mul = s4->delta_tau_n_sign() ? (-1) : 1;
  REQUIRE(s4->delta_tau_n() == (data[ST4_DELTA_TAU_N_V_IDX].second * mul));
  REQUIRE(s4->e_n() == data[ST4_E_N_IDX].second);
  REQUIRE(s4->not_used_1() == data[ST4_NU_1_IDX].second);
  REQUIRE(s4->p4() == data[ST4_P4_IDX].second);
  REQUIRE(s4->f_t() == data[ST4_F_T_IDX].second);
  REQUIRE(s4->not_used_2() == data[ST4_NU_2_IDX].second);
  REQUIRE(s4->n_t() == data[ST4_N_T_IDX].second);
  REQUIRE(s4->n() == data[ST4_N_IDX].second);
  REQUIRE(s4->m() == data[ST4_M_IDX].second);
}

TEST_CASE("parse_string_number_5"){
  string_data data = generate_string_data(5);
  std::string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);
  REQUIRE(gl_string.hamming_code() == data[ST5_HC_OFF + HC_IDX].second);
  REQUIRE(gl_string.pad_1() == data[ST5_HC_OFF + PAD1_IDX].second);
  REQUIRE(gl_string.superframe_number() == data[ST5_HC_OFF + SUPERFRAME_IDX].second);
  REQUIRE(gl_string.pad_2() == data[ST5_HC_OFF + PAD2_IDX].second);
  REQUIRE(gl_string.frame_number() == data[ST5_HC_OFF + FRAME_IDX].second);

  kaitai::kstream str5(inp_data);
  glonass_t str5_data(&str5);
  glonass_t::string_5_t* s5 = static_cast<glonass_t::string_5_t*>(str5_data.data());

  REQUIRE(s5->n_a() == data[ST5_N_A_IDX].second);
  REQUIRE(s5->tau_c() == data[ST5_TAU_C_IDX].second);
  REQUIRE(s5->not_used() == data[ST5_NU_IDX].second);
  REQUIRE(s5->n_4() == data[ST5_N_4_IDX].second);
  REQUIRE(s5->tau_gps() == data[ST5_TAU_GPS_IDX].second);
  REQUIRE(s5->l_n() == data[ST5_L_N_IDX].second);
}

TEST_CASE("parse_string_number_NI"){
  string_data data = generate_string_data((rand() % 10) +  6);
  std::string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);
  REQUIRE(gl_string.hamming_code() == data[ST6_HC_OFF + HC_IDX].second);
  REQUIRE(gl_string.pad_1() == data[ST6_HC_OFF + PAD1_IDX].second);
  REQUIRE(gl_string.superframe_number() == data[ST6_HC_OFF + SUPERFRAME_IDX].second);
  REQUIRE(gl_string.pad_2() == data[ST6_HC_OFF + PAD2_IDX].second);
  REQUIRE(gl_string.frame_number() == data[ST6_HC_OFF + FRAME_IDX].second);

  kaitai::kstream strni(inp_data);
  glonass_t strni_data(&strni);
  glonass_t::string_non_immediate_t* sni = static_cast<glonass_t::string_non_immediate_t*>(strni_data.data());

  REQUIRE(sni->data_1() == data[ST6_DATA_1_IDX].second);
  REQUIRE(sni->data_2() == data[ST6_DATA_2_IDX].second);
}
