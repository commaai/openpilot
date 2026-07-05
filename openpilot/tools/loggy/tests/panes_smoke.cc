#include "catch2/catch.hpp"

#include "tools/loggy/panes/browser.h"
#include "tools/loggy/panes/binary.h"
#include "tools/loggy/panes/historylog.h"
#include "tools/loggy/panes/logs.h"
#include "tools/loggy/panes/map.h"
#include "tools/loggy/panes/messages.h"
#include "tools/loggy/panes/plot.h"
#include "tools/loggy/panes/signal.h"
#include "tools/loggy/panes/dbc.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;

struct TempDir {
  TempDir() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    path = fs::temp_directory_path() / ("loggy_panes_smoke_" + std::to_string(now));
    fs::create_directories(path);
  }

  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }

  fs::path path;
};

void write_text(const fs::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
}

loggy::StoreBatch makeBatch() {
  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 3.0}};
  batch.series.push_back({
    .path = "/carState/vEgo",
    .range = {0.0, 3.0},
    .points = {{0.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/latitude",
    .range = {0.0, 3.0},
    .points = {{0.0, 37.0000}, {1.0, 37.0005}, {2.0, 37.0010}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/longitude",
    .range = {0.0, 3.0},
    .points = {{0.0, -122.0000}, {1.0, -121.9995}, {2.0, -121.9990}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/hasFix",
    .range = {0.0, 3.0},
    .points = {{0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/bearingDeg",
    .range = {0.0, 3.0},
    .points = {{0.0, 90.0}, {1.0, 91.0}, {2.0, 92.0}},
    .segment = 0,
  });
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
  return batch;
}

}  // namespace

TEST_CASE("plot pane helpers parse state and prepare store series") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const auto requests = loggy::parse_plot_series_requests(R"({"series":[{"path":"/carState/vEgo","label":"speed"}]})");
  REQUIRE(requests.size() == 1);
  CHECK(requests[0].path == "/carState/vEgo");
  CHECK(requests[0].label == "speed");

  const auto prepared = loggy::prepare_plot_series(store, requests, {0.0, 3.0}, 1.5, 128);
  REQUIRE(prepared.size() == 1);
  REQUIRE(prepared[0].xs.size() == 3);
  CHECK(prepared[0].has_tracker_value);
  CHECK(prepared[0].tracker_value == 2.0);
}

TEST_CASE("messages pane helper summarizes selected CAN id") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const loggy::MessageId id = loggy::parse_message_id_state(R"({"id":"0:123"})");
  CHECK(loggy::initial_message_id_for_store(store, "{}") == id);
  const loggy::MessageSummary summary = loggy::summarize_message_events(store, id, {0.0, 3.0});
  CHECK(summary.count == 3);
  CHECK(summary.frequency_hz == 1.0);
  REQUIRE(summary.latest_data.size() == 2);
  CHECK(summary.latest_data[0] == 0x03);

  loggy::MessageTableState table_state;
  const auto all_rows = loggy::prepare_message_table_rows(store, {0.0, 3.0}, table_state);
  REQUIRE(all_rows.size() == 2);
  CHECK(all_rows[0].id == id);
  CHECK(all_rows[1].id == loggy::MessageId{.source = 1, .address = 0x456});
  CHECK(all_rows[1].summary.count == 3);
  CHECK(all_rows[1].summary.latest_data.back() == 0x31);

  table_state.filter = "456";
  const auto filtered_rows = loggy::prepare_message_table_rows(store, {0.0, 3.0}, table_state);
  REQUIRE(filtered_rows.size() == 1);
  CHECK(filtered_rows[0].id == loggy::MessageId{.source = 1, .address = 0x456});

  table_state.filter.clear();
  table_state.bus_filter = 1;
  const auto bus_rows = loggy::prepare_message_table_rows(store, {0.0, 3.0}, table_state);
  REQUIRE(bus_rows.size() == 1);
  CHECK(bus_rows[0].id.source == 1);
}

TEST_CASE("signal pane helpers show bit candidates and DBC decoded values") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const loggy::MessageId id{.source = 0, .address = 0x123};
  const loggy::SignalPaneState state;
  const auto bit_rows = loggy::prepare_signal_pane_rows(store, id, {0.0, 3.0}, state);
  REQUIRE(bit_rows.size() == 16);
  const auto bit1 = std::find_if(bit_rows.begin(), bit_rows.end(), [](const auto &row) {
    return row.name == "byte0.bit1";
  });
  REQUIRE(bit1 != bit_rows.end());
  CHECK(bit1->value == "1");
  CHECK(bit1->flip_count == 1);
  CHECK_FALSE(bit1->from_dbc);

  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SOURCE_ALL, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ flag : 8|1@1+ (1,0) [0|1] "" XXX
)", &error));
  const auto dbc_rows = loggy::prepare_signal_pane_rows(store, id, {0.0, 3.0}, state, manager.msg(id));
  REQUIRE(dbc_rows.size() == 2);
  const auto speed = std::find_if(dbc_rows.begin(), dbc_rows.end(), [](const auto &row) {
    return row.name == "speed";
  });
  REQUIRE(speed != dbc_rows.end());
  CHECK(speed->from_dbc);
  CHECK(speed->value == "3 kph");
  CHECK(speed->start_bit == 0);
  CHECK(speed->size == 8);
}

TEST_CASE("history log helpers show hex rows and decoded values") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const loggy::MessageId id{.source = 0, .address = 0x123};
  const loggy::MessageId saved_id{.source = 1, .address = 0x456};
  loggy::HistoryLogState saved_state;
  saved_state.filter = "SPEED";
  saved_state.compare_signal = "speed";
  saved_state.compare_op = ">=";
  saved_state.compare_value = 1.5;
  saved_state.compare_enabled = true;
  saved_state.max_rows = 99;
  saved_state.page_size = 32;
  saved_state.page_index = 2;
  saved_state.export_path = "/tmp/history.csv";
  saved_state.export_status = "Saved /tmp/history.csv";
  const std::string saved_json = loggy::history_log_state_json(saved_id, saved_state);
  CHECK(loggy::parse_message_id_state(saved_json) == saved_id);
  CHECK(loggy::parse_history_log_state(saved_json).filter == "SPEED");
  CHECK(loggy::parse_history_log_state(saved_json).compare_signal == "speed");
  CHECK(loggy::parse_history_log_state(saved_json).compare_op == ">=");
  CHECK(loggy::parse_history_log_state(saved_json).compare_value == 1.5);
  CHECK(loggy::parse_history_log_state(saved_json).compare_enabled);
  CHECK(loggy::parse_history_log_state(saved_json).max_rows == 99);
  CHECK(loggy::parse_history_log_state(saved_json).page_size == 32);
  CHECK(loggy::parse_history_log_state(saved_json).page_index == 2);
  CHECK(loggy::parse_history_log_state(saved_json).export_path == "/tmp/history.csv");
  CHECK(loggy::parse_history_log_state(saved_json).export_status == "Saved /tmp/history.csv");
  CHECK(loggy::parse_history_log_state("not json").max_rows == 1000);
  CHECK(loggy::parse_history_log_state(R"({"max_rows":1})").max_rows == 16);
  CHECK(loggy::parse_history_log_state(R"({"max_rows":999999})").max_rows == 20000);
  CHECK(loggy::parse_history_log_state(R"({"page_size":0})").page_size == 1);
  CHECK(loggy::parse_history_log_state(R"({"page_size":999999})").page_size == 5000);
  CHECK(loggy::parse_history_log_state(R"({"compare_op":"bad"})").compare_op == ">");

  loggy::HistoryLogState state;
  const auto rows = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, state);
  REQUIRE(rows.size() == 3);
  CHECK(rows[0].data_hex == "03 70");
  CHECK(rows[2].data_hex == "00 F0");
  CHECK(rows[0].byte_count == 2);
  CHECK(rows[0].bus_time == 12);

  loggy::HistoryLogState page_state;
  page_state.page_size = 2;
  auto page = loggy::prepare_history_log_page(store, id, {0.0, 3.0}, page_state);
  REQUIRE(page.rows.size() == 2);
  CHECK(page.total_rows == 3);
  CHECK(page.page_count == 2);
  CHECK(page.page_index == 0);
  CHECK(page.rows[0].mono_time == 2.0);
  CHECK(page.rows[1].mono_time == 1.0);
  page_state.page_index = 1;
  page = loggy::prepare_history_log_page(store, id, {0.0, 3.0}, page_state);
  REQUIRE(page.rows.size() == 1);
  CHECK(page.page_index == 1);
  CHECK(page.rows[0].mono_time == 0.0);

  loggy::HistoryLogState limited_state;
  limited_state.max_rows = 2;
  const auto limited = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, limited_state);
  REQUIRE(limited.size() == 2);
  CHECK(limited[0].mono_time == 2.0);
  CHECK(limited[1].mono_time == 1.0);

  state.filter = "03 70";
  const auto filtered = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, state);
  REQUIRE(filtered.size() == 1);
  CHECK(filtered[0].mono_time == 2.0);

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
  loggy::HistoryLogState decoded_state;
  const auto decoded = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, decoded_state, manager.msg(id));
  REQUIRE(decoded.size() == 3);
  CHECK(decoded[0].decoded == "speed=3 kph");

  decoded_state.filter = "SPEED=3";
  const auto decoded_filtered = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, decoded_state, manager.msg(id));
  REQUIRE(decoded_filtered.size() == 1);
  CHECK(decoded_filtered[0].decoded == "speed=3 kph");

  loggy::HistoryLogState compare_state;
  compare_state.compare_enabled = true;
  compare_state.compare_signal = "speed";
  compare_state.compare_op = ">";
  compare_state.compare_value = 1.0;
  const auto compared = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, compare_state, manager.msg(id));
  REQUIRE(compared.size() == 1);
  CHECK(compared[0].mono_time == 2.0);
  CHECK(compared[0].decoded == "speed=3 kph");

  compare_state.compare_op = "<=";
  const auto compared_lte = loggy::prepare_history_log_rows(store, id, {0.0, 3.0}, compare_state, manager.msg(id));
  REQUIRE(compared_lte.size() == 2);
  CHECK(compared_lte[0].mono_time == 1.0);
  CHECK(compared_lte[1].mono_time == 0.0);
}

TEST_CASE("DBC pane helpers parse sources and summarize loaded files") {
  loggy::SourceSet sources;
  std::string error;
  REQUIRE(loggy::parse_dbc_source_set("all", &sources, &error));
  CHECK(sources == loggy::SOURCE_ALL);
  CHECK(error.empty());

  REQUIRE(loggy::parse_dbc_source_set("0, 2 3", &sources, &error));
  const loggy::SourceSet expected_sources{0, 2, 3};
  CHECK(sources == expected_sources);
  CHECK(error.empty());

  REQUIRE_FALSE(loggy::parse_dbc_source_set("bad", &sources, &error));
  CHECK(error == "invalid source: bad");
  REQUIRE_FALSE(loggy::parse_dbc_source_set("999", &sources, &error));
  CHECK(error == "invalid source: 999");

  loggy::DbcPaneState state;
  state.path = "/tmp/demo.dbc";
  state.save_as_path = "/tmp/out.dbc";
  state.sources = "0,1";
  state.opendbc_root = "/tmp/opendbc";
  state.opendbc_filter = "ford";
  state.status = "Loaded";
  const std::string json = loggy::dbc_pane_state_json(state);
  CHECK(loggy::parse_dbc_pane_state(json).path == "/tmp/demo.dbc");
  CHECK(loggy::parse_dbc_pane_state(json).save_as_path == "/tmp/out.dbc");
  CHECK(loggy::parse_dbc_pane_state(json).sources == "0,1");
  CHECK(loggy::parse_dbc_pane_state(json).opendbc_root == "/tmp/opendbc");
  CHECK(loggy::parse_dbc_pane_state(json).opendbc_filter == "ford");
  CHECK(loggy::parse_dbc_pane_state(json).status == "Loaded");

  loggy::DBCManager manager;
  REQUIRE(manager.open(loggy::SourceSet{0, 1}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ flag : 8|1@1+ (1,0) [0|1] "" XXX
)", &error));
  const auto rows = loggy::prepare_dbc_file_rows(manager);
  REQUIRE(rows.size() == 1);
  CHECK(rows[0].name == "inline");
  CHECK(rows[0].sources == "0, 1");
  CHECK(rows[0].message_count == 1);
  CHECK(rows[0].signal_count == 2);
  CHECK(loggy::dbc_file_for_sources(manager, loggy::SourceSet{0}) != nullptr);
  CHECK(loggy::dbc_file_for_sources(manager, loggy::SOURCE_ALL) != nullptr);

  loggy::DBCManager scratch;
  REQUIRE(loggy::create_empty_dbc(scratch, loggy::SOURCE_ALL, "", &error));
  REQUIRE(scratch.findDBCFile(0) != nullptr);
  CHECK(scratch.findDBCFile(0)->name() == "untitled");
  CHECK(scratch.findDBCFile(0)->filename.empty());

  const std::string copied = loggy::dbc_clipboard_text_for_sources(manager, loggy::SourceSet{0}, &error);
  CHECK(error.empty());
  CHECK(copied.find("BO_ 291 TEST_MSG") != std::string::npos);
  REQUIRE(loggy::open_dbc_from_clipboard_text(scratch, loggy::SourceSet{2}, "clipboard", copied, &error));
  REQUIRE(scratch.findDBCFile(2) != nullptr);
  CHECK(scratch.findDBCFile(2)->name() == "clipboard");
  CHECK(scratch.findDBCFile(2)->msg(291) != nullptr);

  REQUIRE_FALSE(loggy::open_dbc_from_clipboard_text(scratch, loggy::SourceSet{3}, "empty", "  \n", &error));
  CHECK(error == "clipboard is empty");
}

TEST_CASE("DBC pane helpers scan opendbc roots with filtering") {
  TempDir temp;
  write_text(temp.path / "ford_lincoln_base_pt.dbc", "VERSION \"\"\n");
  write_text(temp.path / "honda_civic.dbc", "VERSION \"\"\n");
  write_text(temp.path / "notes.txt", "not a dbc");
  fs::create_directories(temp.path / "nested");
  write_text(temp.path / "nested" / "toyota.dbc", "VERSION \"\"\n");

  std::string error;
  auto rows = loggy::prepare_opendbc_file_rows(temp.path, "", 100, &error);
  REQUIRE(rows.size() == 2);
  CHECK(error.empty());
  CHECK(rows[0].name == "ford_lincoln_base_pt");
  CHECK(rows[1].name == "honda_civic");

  rows = loggy::prepare_opendbc_file_rows(temp.path, "LINCOLN", 100, &error);
  REQUIRE(rows.size() == 1);
  CHECK(rows[0].path.find("ford_lincoln_base_pt.dbc") != std::string::npos);

  rows = loggy::prepare_opendbc_file_rows(temp.path, "", 1, &error);
  REQUIRE(rows.size() == 1);

  rows = loggy::prepare_opendbc_file_rows(temp.path / "missing", "", 100, &error);
  CHECK(rows.empty());
  CHECK(error.find("opendbc root not found") != std::string::npos);
}

TEST_CASE("browser pane helpers filter loaded series and plot accepts paths") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const std::vector<std::string> paths = store.seriesPaths();
  REQUIRE(paths.size() == 5);
  CHECK(std::binary_search(paths.begin(), paths.end(), "/carState/vEgo"));

  loggy::BrowserState browser_state;
  auto rows = loggy::prepare_browser_series_rows(store, browser_state);
  REQUIRE(rows.size() == 5);
  CHECK(std::any_of(rows.begin(), rows.end(), [](const auto &row) {
    return row.path == "/carState/vEgo" && row.label == "vEgo";
  }));

  browser_state.filter = "ego";
  rows = loggy::prepare_browser_series_rows(store, browser_state);
  REQUIRE(rows.size() == 1);
  browser_state.filter = "missing";
  rows = loggy::prepare_browser_series_rows(store, browser_state);
  CHECK(rows.empty());

  const std::string state = loggy::plot_state_with_added_series(
    R"({"series":[{"path":"/carState/vEgo","label":"v"}],"max_points":128})",
    "/carState/aEgo");
  const auto requests = loggy::parse_plot_series_requests(state);
  REQUIRE(requests.size() == 2);
  CHECK(requests[0].path == "/carState/vEgo");
  CHECK(requests[1].path == "/carState/aEgo");
  CHECK(loggy::parse_plot_max_points(state, 0) == 128);

  const std::string duplicate_state = loggy::plot_state_with_added_series(state, "/carState/aEgo");
  CHECK(loggy::parse_plot_series_requests(duplicate_state).size() == 2);
}

TEST_CASE("map pane helpers prepare GPS trace and tracker point") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const loggy::MapState state;
  const loggy::MapTrace trace = loggy::prepare_map_trace(store, {0.0, 3.0}, state);
  REQUIRE(trace.valid());
  REQUIRE(trace.points.size() == 3);
  CHECK(trace.min_lat == 37.0000);
  CHECK(trace.max_lat == 37.0010);
  CHECK(trace.min_lon == -122.0000);
  CHECK(trace.max_lon == -121.9990);

  const auto tracker = loggy::map_trace_point_at_time(trace, 1.2);
  REQUIRE(tracker.has_value());
  CHECK(tracker->lat == 37.0005);
  CHECK(tracker->bearing_deg == 91.0);
}

TEST_CASE("binary pane helper builds bit grid and flip counts") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const auto grid = loggy::build_binary_grid(store, loggy::MessageId{.source = 0, .address = 0x123}, {0.0, 3.0});
  REQUIRE(grid.has_value());
  REQUIRE(grid->rows.size() == 2);
  CHECK(grid->rows[0][6].flip_count == 1);
  CHECK(grid->rows[0][7].flip_count == 1);
  CHECK(grid->rows[1][0].value == 0);
  CHECK(grid->max_flip_count == 1);
}

TEST_CASE("logs pane helper filters by text and level") {
  std::vector<loggy::LogEntry> logs = {
    {.mono_time = 1.0, .level = 20, .source = "controlsd", .message = "startup complete", .origin = loggy::LogOrigin::Log},
    {.mono_time = 2.0, .level = 40, .source = "modeld", .message = "model error", .origin = loggy::LogOrigin::OperatingSystem},
    {.mono_time = 3.0, .level = 30, .source = "alert", .message = "take control", .origin = loggy::LogOrigin::Alert},
  };

  const auto errors = loggy::filter_log_entries(logs, "model", 30, 100);
  REQUIRE(errors.size() == 1);
  CHECK(errors[0] == 1);
  CHECK(std::string(loggy::log_level_label(logs[1].level)) == "ERROR");
  CHECK(std::string(loggy::log_origin_label(logs[2].origin)) == "Alert");

  const auto limited = loggy::filter_log_entries(logs, "", 0, 2);
  REQUIRE(limited.size() == 2);
  CHECK(limited[0] == 0);
  CHECK(limited[1] == 1);
}
