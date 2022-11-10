
#include "catch2/catch.hpp"
#include "tools/cabana/dbcmanager.h"

TEST_CASE("DBCManager::generateDBC") {
  DBCManager dbc_origin(nullptr);
  dbc_origin.open("toyota_new_mc_pt_generated");
  QString dbc_string = dbc_origin.generateDBC();

  DBCManager dbc_from_generated(nullptr);
  dbc_from_generated.open("", dbc_string);

  auto &msgs = dbc_origin.messages();
  auto &new_msgs = dbc_from_generated.messages();
  REQUIRE(msgs.size() == new_msgs.size());
  for (auto &[address, m] : msgs) {
    auto new_m = new_msgs.at(address);
    REQUIRE(m.name == new_m.name);
    REQUIRE(m.address == new_m.address);
    REQUIRE(m.size == new_m.size);
    REQUIRE(m.sigs.size() == new_m.sigs.size());
    auto &sig = m.sigs;
    auto &new_sig = new_m.sigs;
    for (int j = 0; j < sig.size(); ++j) {
      REQUIRE(sig[j].name == new_sig[j].name);
      REQUIRE(sig[j].start_bit == new_sig[j].start_bit);
      REQUIRE(sig[j].msb == new_sig[j].msb);
      REQUIRE(sig[j].lsb == new_sig[j].lsb);
      REQUIRE(sig[j].size == new_sig[j].size);
      REQUIRE(sig[j].is_signed == new_sig[j].is_signed);
      REQUIRE(sig[j].factor == new_sig[j].factor);
      REQUIRE(sig[j].offset == new_sig[j].offset);
      REQUIRE(sig[j].is_little_endian == new_sig[j].is_little_endian);
    }
  }
}
