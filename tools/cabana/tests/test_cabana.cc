
#include "catch2/catch.hpp"
#include "tools/cabana/dbcmanager.h"

TEST_CASE("DBCManager::generateDBC") {
  DBCManager dbc_origin(nullptr);
  dbc_origin.open("toyota_new_mc_pt_generated");
  QString dbc_string = dbc_origin.generateDBC();

  DBCManager dbc_from_generated(nullptr);
  dbc_from_generated.open("", dbc_string);

  auto dbc = dbc_origin.getDBC();
  auto new_dbc = dbc_from_generated.getDBC();
  REQUIRE(dbc->msgs.size() == new_dbc->msgs.size());
  for (int i = 0; i < dbc->msgs.size(); ++i) {
    REQUIRE(dbc->msgs[i].name == new_dbc->msgs[i].name);
    REQUIRE(dbc->msgs[i].address == new_dbc->msgs[i].address);
    REQUIRE(dbc->msgs[i].size == new_dbc->msgs[i].size);
    REQUIRE(dbc->msgs[i].sigs.size() == new_dbc->msgs[i].sigs.size());
    auto &sig = dbc->msgs[i].sigs;
    auto &new_sig = new_dbc->msgs[i].sigs;
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
