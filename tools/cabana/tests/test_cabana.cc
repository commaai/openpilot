
#include "opendbc/can/common.h"
#undef INFO
#include "catch2/catch.hpp"
#include "tools/replay/logreader.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

// demo route, first segment
const std::string TEST_RLOG_URL = "https://commadata2.blob.core.windows.net/commadata2/4cf7a6ad03080c90/2021-09-29--13-46-36/0/rlog.bz2";

TEST_CASE("DBCFile::generateDBC") {
  QString fn = QString("%1/%2.dbc").arg(OPENDBC_FILE_PATH, "toyota_new_mc_pt_generated");
  DBCFile dbc_origin(fn);
  DBCFile dbc_from_generated("", dbc_origin.generateDBC());

  REQUIRE(dbc_origin.msgCount() == dbc_from_generated.msgCount());
  auto msgs = dbc_origin.getMessages();
  auto new_msgs = dbc_from_generated.getMessages();
  for (auto &[id, m] : msgs) {
    auto &new_m = new_msgs.at(id);
    REQUIRE(m.name == new_m.name);
    REQUIRE(m.size == new_m.size);
    REQUIRE(m.getSignals().size() == new_m.getSignals().size());
    auto sigs = m.getSignals();
    auto new_sigs = new_m.getSignals();
    for (int i = 0; i < sigs.size(); ++i) {
      REQUIRE(*sigs[i] == *new_sigs[i]);
    }
  }
}

TEST_CASE("Parse can messages") {
  DBCManager dbc(nullptr);
  dbc.open({0}, "toyota_new_mc_pt_generated");
  CANParser can_parser(0, "toyota_new_mc_pt_generated", {}, {});

  LogReader log;
  REQUIRE(log.load(TEST_RLOG_URL, nullptr, {}, true));
  REQUIRE(log.events.size() > 0);
  for (auto e : log.events) {
    if (e->which == cereal::Event::Which::CAN) {
      std::map<std::pair<uint32_t, QString>, std::vector<double>> values_1;
      for (const auto &c : e->event.getCan()) {
        const auto msg = dbc.msg({.source = c.getSrc(), .address = c.getAddress()});
        if (c.getSrc() == 0 && msg) {
          for (auto sig : msg->getSignals()) {
            double val = get_raw_value((uint8_t *)c.getDat().begin(), c.getDat().size(), *sig);
            values_1[{c.getAddress(), sig->name}].push_back(val);
          }
        }
      }

      can_parser.UpdateCans(e->mono_time, e->event.getCan());
      std::vector<SignalValue> values_2;
      can_parser.query_latest(values_2);
      for (auto &[key, v1] : values_1) {
        bool found = false;
        for (auto &v2 : values_2) {
          if (v2.address == key.first && key.second == v2.name.c_str()) {
            REQUIRE(v2.all_values.size() == v1.size());
            REQUIRE(v2.all_values == v1);
            found = true;
            break;
          }
        }
        REQUIRE(found);
      }
    }
  }
}
