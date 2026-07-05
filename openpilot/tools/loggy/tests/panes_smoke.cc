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
#include "tools/loggy/panes/find_bits.h"
#include "tools/loggy/panes/find_signal.h"
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

loggy::StoreBatch makeFindBitsBatch() {
  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 4.0}};
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 0, .address = 0x100},
    .range = {0.0, 4.0},
    .events = {
      {.mono_time = 0.0, .bus_time = 1, .data = {0x00}},
      {.mono_time = 1.0, .bus_time = 2, .data = {0x80}},
      {.mono_time = 2.0, .bus_time = 3, .data = {0x80}},
      {.mono_time = 3.0, .bus_time = 4, .data = {0x00}},
    },
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 0, .address = 0x200},
    .range = {0.0, 4.0},
    .events = {
      {.mono_time = 0.0, .bus_time = 5, .data = {0x00}},
      {.mono_time = 1.0, .bus_time = 6, .data = {0x80}},
      {.mono_time = 2.0, .bus_time = 7, .data = {0x80}},
      {.mono_time = 3.0, .bus_time = 8, .data = {0x00}},
    },
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 0, .address = 0x201},
    .range = {0.0, 4.0},
    .events = {
      {.mono_time = 0.0, .bus_time = 9, .data = {0x80}},
      {.mono_time = 1.0, .bus_time = 10, .data = {0x00}},
      {.mono_time = 2.0, .bus_time = 11, .data = {0x00}},
      {.mono_time = 3.0, .bus_time = 12, .data = {0x80}},
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

  const std::string display_state = loggy::plot_state_with_display_options(
    R"({"series":[{"path":"/carState/vEgo","label":"speed"}],"max_points":128})",
    loggy::PlotSeriesStyle::Scatter,
    loggy::PlotYLimits{.min_enabled = true, .max_enabled = true, .min = -1.0, .max = 4.0});
  CHECK(loggy::parse_plot_series_style(display_state) == loggy::PlotSeriesStyle::Scatter);
  const loggy::PlotYLimits parsed_limits = loggy::parse_plot_y_limits(display_state);
  CHECK(parsed_limits.min_enabled);
  CHECK(parsed_limits.max_enabled);
  CHECK(parsed_limits.min == -1.0);
  CHECK(parsed_limits.max == 4.0);
  CHECK(loggy::parse_plot_max_points(display_state, 0) == 128);
  CHECK(loggy::parse_plot_series_style(R"({"stairs":true})") == loggy::PlotSeriesStyle::Step);

  const auto prepared = loggy::prepare_plot_series(store, requests, {0.0, 3.0}, 1.5, 128);
  REQUIRE(prepared.size() == 1);
  REQUIRE(prepared[0].xs.size() == 3);
  CHECK(prepared[0].has_tracker_value);
  CHECK(prepared[0].tracker_value == 2.0);
  CHECK(loggy::plot_effective_series_style(prepared[0], loggy::PlotSeriesStyle::Auto) == loggy::PlotSeriesStyle::Step);
  CHECK(loggy::plot_effective_series_style(prepared[0], loggy::PlotSeriesStyle::Line) == loggy::PlotSeriesStyle::Line);
  CHECK(loggy::plot_effective_series_style(prepared[0], loggy::PlotSeriesStyle::Scatter) == loggy::PlotSeriesStyle::Scatter);

  const auto auto_bounds = loggy::compute_plot_y_axis_bounds(prepared, {});
  CHECK_FALSE(auto_bounds.active);
  CHECK(auto_bounds.min == Approx(0.9));
  CHECK(auto_bounds.max == Approx(3.1));
  loggy::PlotYLimits min_only;
  min_only.min_enabled = true;
  min_only.min = 0.0;
  const auto min_bounds = loggy::compute_plot_y_axis_bounds(prepared, min_only);
  CHECK(min_bounds.active);
  CHECK(min_bounds.min == 0.0);
  CHECK(min_bounds.max == Approx(3.1));
  loggy::PlotYLimits max_only;
  max_only.max_enabled = true;
  max_only.max = 5.0;
  const auto max_bounds = loggy::compute_plot_y_axis_bounds(prepared, max_only);
  CHECK(max_bounds.active);
  CHECK(max_bounds.min == Approx(0.9));
  CHECK(max_bounds.max == 5.0);
  loggy::PlotYLimits degenerate;
  degenerate.min_enabled = true;
  degenerate.max_enabled = true;
  degenerate.min = 5.0;
  degenerate.max = 5.0;
  const auto degenerate_bounds = loggy::compute_plot_y_axis_bounds(prepared, degenerate);
  CHECK(degenerate_bounds.min == 4.5);
  CHECK(degenerate_bounds.max == 5.5);
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
  const auto binary_grid = loggy::build_binary_grid(store, id, {0.0, 3.0});
  REQUIRE(binary_grid.has_value());
  REQUIRE(binary_grid->rows.size() == 2);
  CHECK(binary_grid->rows[0][6].value == 1);
  CHECK(binary_grid->rows[0][6].flip_count == 1);
  loggy::Signal draft_signal;
  REQUIRE(loggy::binary_signal_from_bit_range(1, 6, &draft_signal));
  CHECK(draft_signal.start_bit == 1);
  CHECK(draft_signal.size == 6);
  CHECK(draft_signal.is_little_endian);
  CHECK(draft_signal.max == 63.0);

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

TEST_CASE("analysis helpers find candidate signals and similar bits") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  loggy::FindSignalParams signal_params;
  signal_params.range = {0.0, 3.0};
  signal_params.buses = {0};
  signal_params.addresses = {0x123};
  signal_params.min_size = 2;
  signal_params.max_size = 2;
  signal_params.compare = loggy::FindSignalCompare::Any;
  const auto candidates = loggy::prepare_find_signal_candidates(store, signal_params);
  const auto bit0_2 = std::find_if(candidates.begin(), candidates.end(), [](const loggy::FindSignalResult &result) {
    return result.id == loggy::MessageId{.source = 0, .address = 0x123} &&
           result.sig.start_bit == 0 && result.sig.size == 2;
  });
  REQUIRE(bit0_2 != candidates.end());
  REQUIRE(bit0_2->matches.size() == 3);
  CHECK(bit0_2->matches[0] == std::pair<double, double>{0.0, 0.0});
  CHECK(bit0_2->matches[1] == std::pair<double, double>{1.0, 1.0});
  CHECK(bit0_2->matches[2] == std::pair<double, double>{2.0, 3.0});

  signal_params.compare = loggy::FindSignalCompare::Equal;
  signal_params.target_value = 1.0;
  const auto equal_one = loggy::prepare_find_signal_candidates(store, signal_params);
  const auto one_hit = std::find_if(equal_one.begin(), equal_one.end(), [](const loggy::FindSignalResult &result) {
    return result.sig.start_bit == 0 && result.sig.size == 2;
  });
  REQUIRE(one_hit != equal_one.end());
  REQUIRE(one_hit->matches.size() == 1);
  CHECK(one_hit->matches[0] == std::pair<double, double>{1.0, 1.0});

  signal_params.target_value = 3.0;
  const auto equal_three = loggy::prepare_find_signal_candidates(store, signal_params);
  const auto three_hit = std::find_if(equal_three.begin(), equal_three.end(), [](const loggy::FindSignalResult &result) {
    return result.sig.start_bit == 0 && result.sig.size == 2;
  });
  REQUIRE(three_hit != equal_three.end());
  REQUIRE(three_hit->matches.size() == 1);
  CHECK(three_hit->matches[0] == std::pair<double, double>{2.0, 3.0});

  loggy::dbc()->closeAll();
  TempDir temp;
  loggy::SessionConfig config;
  config.settings_path = (temp.path / "settings.json").string();
  loggy::Session session(config);
  std::string error;
  REQUIRE(loggy::dbc()->open(loggy::SOURCE_ALL, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
)", &error));
  REQUIRE(loggy::commit_find_signal_result(session, "analysis", *bit0_2, &error));
  loggy::Msg *created_msg = loggy::dbc()->msg(loggy::MessageId{.source = 0, .address = 0x123});
  REQUIRE(created_msg != nullptr);
  REQUIRE(created_msg->sig("NEW_SIGNAL_1") != nullptr);
  CHECK(created_msg->sig("NEW_SIGNAL_1")->start_bit == 0);
  CHECK(created_msg->sig("NEW_SIGNAL_1")->size == 2);
  CHECK(session.selection("analysis").has_selected_msg);
  CHECK(session.selection("analysis").selected_msg_id == loggy::MessageId{.source = 0, .address = 0x123});
  session.dbc_undo().undo();
  CHECK(loggy::dbc()->msg(loggy::MessageId{.source = 0, .address = 0x123}) == nullptr);
  loggy::dbc()->closeAll();

  loggy::Store bits_store;
  bits_store.stage(makeFindBitsBatch());
  bits_store.beginFrame();
  loggy::FindBitsParams bits_params;
  bits_params.range = {0.0, 4.0};
  bits_params.source_bus = 0;
  bits_params.source_address = 0x100;
  bits_params.byte_idx = 0;
  bits_params.bit_idx = 0;
  bits_params.find_bus = 0;
  bits_params.equal = true;
  bits_params.min_msgs = 0;
  const auto events = loggy::collect_find_bits_events(bits_store, bits_params);
  REQUIRE(events.size() == 8);
  const auto rows = loggy::scan_find_bits_events(events, bits_params);
  REQUIRE_FALSE(rows.empty());
  CHECK(rows[0].address == 0x200);
  CHECK(rows[0].byte_idx == 0);
  CHECK(rows[0].bit_idx == 0);
  CHECK(rows[0].mismatches == 0);
  CHECK(rows[0].total == 4);
  CHECK(rows[0].percent == 0.0f);

  bits_params.min_msgs = 4;
  CHECK(loggy::scan_find_bits_events(events, bits_params).empty());

  loggy::SessionConfig bits_config;
  bits_config.settings_path = (temp.path / "bits_settings.json").string();
  loggy::Session bits_session(bits_config);
  loggy::activate_find_bits_row(bits_session, "analysis", rows[0], 0);
  CHECK(bits_session.selection("analysis").has_selected_msg);
  CHECK(bits_session.selection("analysis").selected_msg_id == loggy::MessageId{.source = 0, .address = 0x200});
}

TEST_CASE("signal pane helpers show bit candidates and DBC decoded values") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const loggy::MessageId id{.source = 0, .address = 0x123};
  const loggy::SignalPaneState state;
  loggy::SignalPaneState saved_signal_state;
  saved_signal_state.filter = "speed";
  saved_signal_state.selected_signal = "speed";
  saved_signal_state.sparkline_seconds = 7;
  const std::string signal_state_json = loggy::signal_pane_state_json(saved_signal_state);
  CHECK(loggy::parse_signal_pane_state(signal_state_json).filter == "speed");
  CHECK(loggy::parse_signal_pane_state(signal_state_json).selected_signal == "speed");
  CHECK(loggy::parse_signal_pane_state(signal_state_json).sparkline_seconds == 7);
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
  REQUIRE(speed->sparkline.values.size() == 3);
  CHECK(speed->sparkline.values[0] == 0.0);
  CHECK(speed->sparkline.values[1] == 1.0);
  CHECK(speed->sparkline.values[2] == 3.0);
  CHECK(speed->sparkline.min == 0.0);
  CHECK(speed->sparkline.max == 3.0);

  loggy::UndoStack undo;
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->sig("speed") != nullptr);
  loggy::SignalEditModel edit = loggy::signal_edit_model_from_signal(*msg->sig("speed"));
  edit.name = "speed_mps";
  edit.factor = 0.5;
  edit.unit = "m/s";
  CHECK(loggy::signal_edit_model_changed(edit, *msg->sig("speed")));
  REQUIRE(loggy::apply_signal_edit(undo, manager, id, edit, &error));
  CHECK(error.empty());
  CHECK(undo.count() == 1);
  CHECK(msg->sig("speed") == nullptr);
  REQUIRE(msg->sig("speed_mps") != nullptr);
  CHECK(msg->sig("speed_mps")->factor == 0.5);
  undo.undo();
  REQUIRE(msg->sig("speed") != nullptr);
  CHECK(msg->sig("speed_mps") == nullptr);
  undo.redo();
  REQUIRE(msg->sig("speed_mps") != nullptr);
  CHECK(msg->sig("speed_mps")->unit == "m/s");

  loggy::ValueDescription parsed_desc;
  REQUIRE(loggy::parse_signal_value_descriptions("0 \"stopped\" 6 \"cruise\"", &parsed_desc, &error));
  CHECK(error.empty());
  CHECK(parsed_desc == loggy::ValueDescription{{0.0, "stopped"}, {6.0, "cruise"}});
  CHECK(loggy::signal_value_descriptions_text(parsed_desc) == "0 \"stopped\" 6 \"cruise\"");
  CHECK_FALSE(loggy::parse_signal_value_descriptions("0 stopped", &parsed_desc, &error));
  CHECK(error == "value description entry needs a quoted description");

  edit = loggy::signal_edit_model_from_signal(*msg->sig("speed_mps"));
  edit.val_desc = {{0.0, "stopped"}, {6.0, "cruise"}};
  REQUIRE(loggy::apply_signal_edit(undo, manager, id, edit, &error));
  CHECK(error.empty());
  REQUIRE(msg->sig("speed_mps") != nullptr);
  CHECK(msg->sig("speed_mps")->formatValue(3.0) == "cruise");
  undo.undo();
  REQUIRE(msg->sig("speed_mps") != nullptr);
  CHECK(msg->sig("speed_mps")->val_desc.empty());
  undo.redo();
  REQUIRE(msg->sig("speed_mps") != nullptr);
  CHECK(msg->sig("speed_mps")->val_desc == loggy::ValueDescription{{0.0, "stopped"}, {6.0, "cruise"}});
  REQUIRE(loggy::remove_signal_edit(undo, manager, id, "speed_mps", &error));
  CHECK(error.empty());
  CHECK(msg->sig("speed_mps") == nullptr);
  REQUIRE(msg->sig("flag") != nullptr);
  undo.undo();
  REQUIRE(msg->sig("speed_mps") != nullptr);
  undo.redo();
  CHECK(msg->sig("speed_mps") == nullptr);
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

  loggy::DBCFile *loaded_file = loggy::dbc_file_for_sources(manager, loggy::SourceSet{0});
  loggy::SourceSet reassigned_sources;
  REQUIRE(loggy::assign_dbc_file_sources(manager, loaded_file, "2", &reassigned_sources, &error));
  const loggy::SourceSet source_two{2};
  CHECK(reassigned_sources == source_two);
  CHECK(manager.findDBCFile(0) == nullptr);
  CHECK(manager.findDBCFile(2) == loaded_file);
  CHECK(manager.sources(loaded_file) == source_two);

  TempDir assign_temp;
  const fs::path a_path = assign_temp.path / "a.dbc";
  const fs::path b_path = assign_temp.path / "b.dbc";
  write_text(a_path, R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 A_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
)");
  write_text(b_path, R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 292 B_MSG: 2 XXX
 SG_ flag : 0|1@1+ (1,0) [0|1] "" XXX
)");
  loggy::DBCManager assignment_manager;
  REQUIRE(assignment_manager.open(loggy::SourceSet{0}, a_path.string(), &error));
  REQUIRE(assignment_manager.open(loggy::SourceSet{1, 2}, b_path.string(), &error));
  loggy::LoggySettings settings;
  loggy::set_dbc_assignment(&settings, "0", a_path.string());
  loggy::set_dbc_assignment(&settings, "1, 2", b_path.string());
  loggy::set_dbc_assignment(&settings, "all", "/tmp/fallback.dbc");
  loggy::set_dbc_assignment(&settings, "3", "/tmp/unrelated.dbc");
  loggy::DBCFile *a_file = assignment_manager.findDBCFile(0);
  REQUIRE(loggy::assign_dbc_file_sources(assignment_manager, a_file, "1", &reassigned_sources, &error));
  loggy::sync_dbc_assignments_from_loaded_files(assignment_manager, &settings);
  CHECK(settings.dbc_assignments.count("0") == 0);
  CHECK(settings.dbc_assignments.count("1, 2") == 0);
  CHECK(settings.dbc_assignments.at("1") == a_path.string());
  CHECK(settings.dbc_assignments.at("2") == b_path.string());
  CHECK(settings.dbc_assignments.at("all") == "/tmp/fallback.dbc");
  CHECK(settings.dbc_assignments.at("3") == "/tmp/unrelated.dbc");
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
  browser_state.sparkline_seconds = 1;
  const std::string browser_json = loggy::browser_state_json(browser_state);
  CHECK(loggy::parse_browser_state(browser_json).sparkline_seconds == 1);
  auto rows = loggy::prepare_browser_series_rows(store, browser_state);
  REQUIRE(rows.size() == 5);
  CHECK(std::any_of(rows.begin(), rows.end(), [](const auto &row) {
    return row.path == "/carState/vEgo" && row.label == "vEgo";
  }));
  const auto v_ego_row = std::find_if(rows.begin(), rows.end(), [](const auto &row) {
    return row.path == "/carState/vEgo";
  });
  REQUIRE(v_ego_row != rows.end());
  const loggy::BrowserSeriesRow detailed_v_ego = loggy::enrich_browser_series_row(store, *v_ego_row, {0.0, 3.0}, 1.5, browser_state);
  CHECK(detailed_v_ego.has_value);
  CHECK(detailed_v_ego.value == "2.5");
  REQUIRE(detailed_v_ego.sparkline.values.size() == 2);
  CHECK(detailed_v_ego.sparkline.values[0] == 2.0);
  CHECK(detailed_v_ego.sparkline.values[1] == 3.0);
  CHECK(detailed_v_ego.sparkline.min == 2.0);
  CHECK(detailed_v_ego.sparkline.max == 3.0);

  browser_state.filter = "ego";
  rows = loggy::prepare_browser_series_rows(store, browser_state);
  REQUIRE(rows.size() == 1);
  browser_state.filter = "missing";
  rows = loggy::prepare_browser_series_rows(store, browser_state);
  CHECK(rows.empty());

  const std::string state = loggy::plot_state_with_added_series(
    R"({"series":[{"path":"/carState/vEgo","label":"v"}],"max_points":128,"style":"scatter","y_limits":{"min":-2,"max":8}})",
    "/carState/aEgo");
  const auto requests = loggy::parse_plot_series_requests(state);
  REQUIRE(requests.size() == 2);
  CHECK(requests[0].path == "/carState/vEgo");
  CHECK(requests[1].path == "/carState/aEgo");
  CHECK(loggy::parse_plot_max_points(state, 0) == 128);
  CHECK(loggy::parse_plot_series_style(state) == loggy::PlotSeriesStyle::Scatter);
  const loggy::PlotYLimits state_limits = loggy::parse_plot_y_limits(state);
  CHECK(state_limits.min_enabled);
  CHECK(state_limits.max_enabled);
  CHECK(state_limits.min == -2.0);
  CHECK(state_limits.max == 8.0);

  const std::string duplicate_state = loggy::plot_state_with_added_series(state, "/carState/aEgo");
  CHECK(loggy::parse_plot_series_requests(duplicate_state).size() == 2);
}

TEST_CASE("plot pane helpers apply derivative and scale transforms") {
  loggy::Store store;
  store.stage(makeBatch());
  store.beginFrame();

  const std::string state = R"({
    "series": [
      {"path": "/carState/vEgo", "label": "scaled", "transform": "scale", "scale": -2, "offset": 10, "color": "#123456"},
      {"path": "/carState/vEgo", "label": "dv", "transform": "derivative"},
      {"path": "/carState/vEgo", "label": "dv_fixed", "transform": "derivative", "derivative_dt": 0.5}
    ]
  })";
  const std::vector<loggy::PlotSeriesRequest> requests = loggy::parse_plot_series_requests(state);
  REQUIRE(requests.size() == 3);
  CHECK(requests[0].transform == loggy::PlotSeriesTransform::Scale);
  CHECK(requests[0].scale == -2.0);
  CHECK(requests[0].offset == 10.0);
  CHECK(requests[0].color == "#123456");
  CHECK(requests[1].transform == loggy::PlotSeriesTransform::Derivative);
  CHECK(requests[2].derivative_dt == 0.5);

  const std::string round_trip = loggy::plot_state_with_display_options(state, loggy::PlotSeriesStyle::Line, {});
  const std::vector<loggy::PlotSeriesRequest> round_trip_requests = loggy::parse_plot_series_requests(round_trip);
  REQUIRE(round_trip_requests.size() == 3);
  CHECK(round_trip_requests[0].transform == loggy::PlotSeriesTransform::Scale);
  CHECK(round_trip_requests[0].scale == -2.0);
  CHECK(round_trip_requests[0].offset == 10.0);
  CHECK(round_trip_requests[0].color == "#123456");
  CHECK(round_trip_requests[2].derivative_dt == 0.5);

  const std::vector<loggy::PreparedPlotSeries> series =
    loggy::prepare_plot_series(store, requests, {0.0, 3.0}, 2.0, 100);
  REQUIRE(series.size() == 3);
  REQUIRE(series[0].ys.size() == 3);
  CHECK(series[0].ys[0] == 8.0);
  CHECK(series[0].ys[1] == 6.0);
  CHECK(series[0].ys[2] == 4.0);

  REQUIRE(series[1].xs.size() == 2);
  CHECK(series[1].xs[0] == 1.0);
  CHECK(series[1].xs[1] == 2.0);
  CHECK(series[1].ys[0] == 1.0);
  CHECK(series[1].ys[1] == 1.0);

  REQUIRE(series[2].ys.size() == 2);
  CHECK(series[2].ys[0] == 2.0);
  CHECK(series[2].ys[1] == 2.0);
  CHECK(series[2].has_tracker_value);
  CHECK(series[2].tracker_value == 2.0);
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

  loggy::TimelineModel timeline({0.0, 3.0});
  timeline.set_spans(std::vector<loggy::TimelineSpan>{
    {.start_time = 0.5, .end_time = 1.5, .kind = loggy::TimelineSpanKind::Engaged},
    {.start_time = 1.8, .end_time = 2.2, .kind = loggy::TimelineSpanKind::AlertWarning},
  });
  const loggy::MapTrace classified = loggy::prepare_map_trace(store, {0.0, 3.0}, state, &timeline);
  REQUIRE(classified.points.size() == 3);
  CHECK(classified.points[0].kind == loggy::TimelineSpanKind::None);
  CHECK(classified.points[1].kind == loggy::TimelineSpanKind::Engaged);
  CHECK(classified.points[2].kind == loggy::TimelineSpanKind::AlertWarning);
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
    {.mono_time = 1.0, .boot_time = 101.0, .wall_time = 1001.0, .level = 20, .source = "controlsd", .message = "startup complete", .origin = loggy::LogOrigin::Log},
    {.mono_time = 2.0, .boot_time = 102.0, .wall_time = 1002.0, .level = 40, .source = "modeld", .message = "model error", .origin = loggy::LogOrigin::OperatingSystem},
    {.mono_time = 3.0, .boot_time = 103.0, .wall_time = 1003.0, .level = 30, .source = "alert", .message = "take control", .origin = loggy::LogOrigin::Alert},
  };

  loggy::LogPaneState state;
  state.filter = "model";
  state.source_filter = "modeld";
  state.min_level = 30;
  state.origin_filter = static_cast<int>(loggy::LogOrigin::OperatingSystem);
  state.time_mode = 2;
  state.follow = false;
  state.max_rows = 12;
  const std::string logs_json = loggy::logs_pane_state_json(state);
  CHECK(loggy::parse_logs_pane_state(logs_json).filter == "model");
  CHECK(loggy::parse_logs_pane_state(logs_json).source_filter == "modeld");
  CHECK(loggy::parse_logs_pane_state(logs_json).min_level == 30);
  CHECK(loggy::parse_logs_pane_state(logs_json).origin_filter == static_cast<int>(loggy::LogOrigin::OperatingSystem));
  CHECK(loggy::parse_logs_pane_state(logs_json).time_mode == 2);
  CHECK_FALSE(loggy::parse_logs_pane_state(logs_json).follow);
  CHECK(loggy::parse_logs_pane_state(logs_json).max_rows == 12);

  const auto errors = loggy::filter_log_entries(logs, "model", 30, 100);
  REQUIRE(errors.size() == 1);
  CHECK(errors[0] == 1);
  CHECK(std::string(loggy::log_level_label(logs[1].level)) == "ERROR");
  CHECK(std::string(loggy::log_origin_label(logs[2].origin)) == "Alert");
  CHECK(std::string(loggy::log_time_mode_label(0)) == "Route");
  CHECK(std::string(loggy::log_time_mode_label(1)) == "Boot");
  CHECK(std::string(loggy::log_time_mode_label(2)) == "Wall");
  CHECK(loggy::log_time_text(logs[1], 0) == "2.00");
  CHECK(loggy::log_time_text(logs[1], 1) == "102.00");
  CHECK(loggy::log_time_text(logs[1], 2) == "1002");

  const auto source_os = loggy::filter_log_entries(logs, loggy::LogFilterParams{
    .filter = "",
    .source_filter = "MODEL",
    .min_level = 0,
    .origin_filter = static_cast<int>(loggy::LogOrigin::OperatingSystem),
    .max_rows = 100,
  });
  REQUIRE(source_os.size() == 1);
  CHECK(source_os[0] == 1);

  const auto alerts = loggy::filter_log_entries(logs, loggy::LogFilterParams{
    .filter = "",
    .source_filter = "",
    .min_level = 0,
    .origin_filter = static_cast<int>(loggy::LogOrigin::Alert),
    .max_rows = 100,
  });
  REQUIRE(alerts.size() == 1);
  CHECK(alerts[0] == 2);

  const auto limited = loggy::filter_log_entries(logs, "", 0, 2);
  REQUIRE(limited.size() == 2);
  CHECK(limited[0] == 0);
  CHECK(limited[1] == 1);
}
