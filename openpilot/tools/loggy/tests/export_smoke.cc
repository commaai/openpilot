#include "catch2/catch.hpp"

#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/backend/export.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace {

void populateStore(loggy::Store *store) {
  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 3.0}};
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 0, .address = 0x123},
    .range = {0.0, 3.0},
    .events = {
      {.mono_time = 0.0, .bus_time = 10, .data = {0x00, 0xF0}},
      {.mono_time = 1.0, .bus_time = 11, .data = {0x01, 0xF0}},
      {.mono_time = 2.0, .bus_time = 12, .data = {0x03, 0x70}},
    },
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 1, .address = 0x456},
    .range = {0.5, 2.5},
    .events = {
      {.mono_time = 0.5, .bus_time = 20, .data = {0x10, 0x20, 0x30}},
      {.mono_time = 1.5, .bus_time = 21, .data = {0x11, 0x20, 0x30}},
      {.mono_time = 2.5, .bus_time = 22, .data = {0x12, 0x20, 0x31}},
    },
    .segment = 0,
  });
  store->stage(std::move(batch));
  store->beginFrame();
}

}  // namespace

TEST_CASE("CSV helpers escape cells and export CAN rows") {
  CHECK(loggy::csv_escape("plain") == "plain");
  CHECK(loggy::csv_escape("a,b\"c") == "\"a,b\"\"c\"");

  loggy::Store store;
  populateStore(&store);
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SOURCE_ALL, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
)", &error));
  const loggy::MessageId id{.source = 0, .address = 0x123};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);

  const std::string message_csv = loggy::can_message_csv(store, id, {0.0, 3.0}, msg);
  CHECK(message_csv.find("mono_time,bus_time,bus,address,length,hex,decoded\n") == 0);
  CHECK(message_csv.find("2.000000,12,0,0x123,2,03 70,speed=3 kph\n") != std::string::npos);

  const std::string stream_csv = loggy::can_stream_csv(store, {0.0, 3.0});
  const size_t first = stream_csv.find("0.000000,10,0,0x123,2,00 F0,");
  const size_t second = stream_csv.find("0.500000,20,1,0x456,3,10 20 30,");
  const size_t last = stream_csv.find("2.500000,22,1,0x456,3,12 20 31,");
  REQUIRE(first != std::string::npos);
  REQUIRE(second != std::string::npos);
  REQUIRE(last != std::string::npos);
  CHECK(first < second);
  CHECK(second < last);

  const loggy::Signal *speed = msg->sig("speed");
  REQUIRE(speed != nullptr);
  const std::string signal_csv = loggy::can_signal_csv(store, id, {0.0, 3.0}, *speed);
  CHECK(signal_csv.find("mono_time,bus_time,bus,address,signal,value,unit,hex\n") == 0);
  CHECK(signal_csv.find("2.000000,12,0,0x123,speed,3,kph,03 70\n") != std::string::npos);

  const fs::path out_path = fs::temp_directory_path() / "loggy_export_smoke" / "message.csv";
  std::string write_error = "not cleared";
  REQUIRE(loggy::write_csv_file(out_path, message_csv, &write_error));
  CHECK(write_error.empty());
  std::ifstream in(out_path, std::ios::binary);
  std::stringstream contents;
  contents << in.rdbuf();
  CHECK(contents.str() == message_csv);
  fs::remove_all(out_path.parent_path());

  REQUIRE_FALSE(loggy::write_csv_file({}, message_csv, &write_error));
  CHECK(write_error == "empty export path");
}
