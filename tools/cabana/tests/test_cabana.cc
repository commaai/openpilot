
#include "catch2/catch.hpp"
#include "tools/cabana/dbcmanager.h"

TEST_CASE("DBCManager::generateDBC") {
  DBCManager dbc_origin(nullptr);
  dbc_origin.open("toyota_new_mc_pt_generated");
  DBCManager dbc_from_generated(nullptr);
  dbc_from_generated.open("", dbc_origin.generateDBC());

  auto &msgs = dbc_origin.messages();
  auto &new_msgs = dbc_from_generated.messages();
  REQUIRE(msgs.size() == new_msgs.size());
  for (auto &[address, m] : msgs) {
    auto new_m = new_msgs.at(address);
    REQUIRE(m.name == new_m.name);
    REQUIRE(m.size == new_m.size);
    REQUIRE(m.sigs.size() == new_m.sigs.size());
    for (auto &[name, sig] : m.sigs)
      REQUIRE(sig == new_m.sigs[name]);
  }
}
