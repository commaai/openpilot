
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
    REQUIRE(m.size == new_m.size);
    REQUIRE(m.sigs.size() == new_m.sigs.size());
    for (auto &[name, sig] : m.sigs) {
      auto &new_sig = new_m.sigs[name];
      REQUIRE(sig.name == new_sig.name);
      REQUIRE(sig.start_bit == new_sig.start_bit);
      REQUIRE(sig.msb == new_sig.msb);
      REQUIRE(sig.lsb == new_sig.lsb);
      REQUIRE(sig.size == new_sig.size);
      REQUIRE(sig.is_signed == new_sig.is_signed);
      REQUIRE(sig.factor == new_sig.factor);
      REQUIRE(sig.offset == new_sig.offset);
      REQUIRE(sig.is_little_endian == new_sig.is_little_endian);
    }
  }
}
