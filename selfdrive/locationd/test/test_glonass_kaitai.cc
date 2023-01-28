#include <iostream>
#include <vector>
#include <bitset>
#include <cassert>

#include "catch2/catch.hpp"
#include "../generated/glonass.h"

using namespace std;

typedef vector<pair<int, int64_t>> string_data;

#define IDLE_CHIP_IDX 0
#define STRING_NUMBER_IDX 1

// Indexes for string number 3
#define P3_IDX 2
#define GAMMA_N_IDX 3
#define NU_1_IDX 4
#define P_IDX 5
#define L_N_IDX 6
#define Z_VEL_IDX 7
#define Z_ACCEL_IDX 8
#define Z_IDX 9
#define HC_IDX 10
#define PAD_1_IDX 11
#define SUPERFRAME_IDX 12
#define PAD2_IDX 13
#define FRAEM_IDX 14

string generate_inp_data(string_data& data) {
  string inp_data = "";
  for (auto& [b, v] : data) {
    string tmp = bitset<32>(v).to_string();
    inp_data += tmp.substr(32-b, b);
  }
  assert(inp_data.size() == 128);

  string string_data;
  string_data.reserve(16);
  for (int i = 0; i < 128; i+=8) {
    string substr = inp_data.substr(i, 8);
    string_data.push_back( (uint8_t)stoi(substr.c_str(), 0, 2));
  }

  return string_data;
}

string_data generate_string_data(uint8_t string_number) {
  vector<pair<int, int64_t>> data; //<bit length, value>
  data.push_back({1, 0}); // idle chip
  data.push_back({4, string_number}); // string number

  if (string_number == 3) {
    data.push_back({1, 0}); // p3
    data.push_back({11, -123}); // gamma_n
    data.push_back({1, 0}); // not_used
    data.push_back({2, 2}); // p
    data.push_back({1, 1}); // l_n
    data.push_back({24, -1337}); // z_vel
    data.push_back({5, 9}); // z_accel
    data.push_back({27, -100023}); // z
    data.push_back({8, 1010}); // hamming code
    data.push_back({11, 1234}); // pad
    data.push_back({16, 7331}); // superframe
    data.push_back({8, 13}); // pad
    data.push_back({8, 4}); // frame
  }
  else {
    assert(0 && "string number not implemented");
  }
  return data;
}

TEST_CASE("parse_string_number_3"){
  string_data data = generate_string_data(3);
  string inp_data = generate_inp_data(data);

  kaitai::kstream stream(inp_data);
  glonass_t gl_string(&stream);

  REQUIRE(gl_string.idle_chip() == data[IDLE_CHIP_IDX].second);
  REQUIRE(gl_string.string_number() == data[STRING_NUMBER_IDX].second);

  kaitai::kstream str3(inp_data);
  glonass_t str3_data(&str3);
  glonass_t::string_3_t* s3 = static_cast<glonass_t::string_3_t*>(str3_data.data());

  REQUIRE(s3->p3() == data[P3_IDX].second);
  REQUIRE(s3->gamma_n() == data[GAMMA_N_IDX].second);
  REQUIRE(s3->not_used() == data[NU_1_IDX].second);
  REQUIRE(s3->p() == data[P_IDX].second);
  REQUIRE(s3->l_n() == data[L_N_IDX].second);
  REQUIRE(s3->z_vel() == data[Z_VEL_IDX].second);
  REQUIRE(s3->z_accel() == data[Z_ACCEL_IDX].second);
  REQUIRE(s3->z() == data[Z_IDX].second);
}
