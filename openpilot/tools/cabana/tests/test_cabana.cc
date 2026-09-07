
#include <atomic>
#include <chrono>
#include <cmath>
#include <capnp/message.h>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "tools/replay/py_downloader.h"
#include "tools/replay/logreader.h"

#include "common/tests/native_test.h"
#include "tools/cabana/analysis/logtelemetry.h"
#include "tools/cabana/dbc/dbcfile.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/routes.h"
#include "tools/cabana/ui/qtstate.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/cabana/ui/chart/downsample.h"
#include "tools/cabana/ui/chart/analysis.h"
#include "tools/cabana/ui/chart/layout.h"
#include "tools/cabana/ui/chart/signaltree.h"
#include "tools/cabana/utils/strings.h"

const std::string TEST_RLOG_URL = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";

void test_message_id_parsing() {
  for (const auto &text : {"", "1", ":123", "1:", "-1:123", "256:1", "1:100000000", "1:1junk", "1junk:1", "1:1:1"}) {
    REQUIRE(!MessageId::parse(text));
    REQUIRE(MessageId::fromString(text) == MessageId{});
  }
  const MessageId expected{255, 0xffffffff};
  REQUIRE(MessageId::parse("255:FFFFFFFF") == expected);
  REQUIRE(MessageId::parse("255:ffffffff") == expected);
  REQUIRE(MessageId::parse(expected.toString()) == expected);
  REQUIRE(MessageId::parse("0:0") == MessageId{});
}

void test_generate_dbc() {
  std::string fn = std::string(OPENDBC_FILE_PATH) + "/tesla_can.dbc";
  DBCFile dbc_origin(fn);
  DBCFile dbc_from_generated("", dbc_origin.generateDBC());

  REQUIRE(dbc_origin.getMessages().size() == dbc_from_generated.getMessages().size());
  auto &msgs = dbc_origin.getMessages();
  auto &new_msgs = dbc_from_generated.getMessages();
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

void test_comment_order() {
  // Ensure that message comments are followed by signal comments and in the correct order
  std::string content = R"(BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

BO_ 162 message_2: 8 EON
 SG_ signal_2 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment";
CM_ SG_ 160 signal_1 "signal comment";
CM_ BO_ 162 "message comment";
CM_ SG_ 162 signal_2 "signal comment";
)";
  DBCFile dbc("", content);
  REQUIRE(dbc.generateDBC() == content);
}

void test_preserve_original_header() {
  std::string content = R"(VERSION "1.0"

NS_ :
 CM_

BS_:

BU_: EON

BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment";
CM_ SG_ 160 signal_1 "signal comment";
)";
  DBCFile dbc("", content);
  REQUIRE(dbc.generateDBC() == content);
}

void test_escaped_quotes() {
  std::string content = R"(BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment with \"escaped quotes\"";
CM_ SG_ 160 signal_1 "signal comment with \"escaped quotes\"";
)";
  DBCFile dbc("", content);
  REQUIRE(dbc.generateDBC() == content);
}

void test_parse_dbc() {
  std::string content = R"(
BO_ 160 message_1: 8 EON
  SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit"  XXX
  SG_ signal_2 : 12|1@1+ (1.0,0.0) [0.0|1] ""  XXX

BO_ 162 message_1: 8 XXX
  SG_ signal_1 M : 0|12@1+ (1,0) [0|4095] "unit" XXX
  SG_ signal_2 M4 : 12|1@1+ (1.0,0.0) [0.0|1] "" XXX

VAL_ 160 signal_1 0 "disabled" 1.2 "initializing" 2 "fault";

CM_ BO_ 160 "message comment" ;
CM_ SG_ 160 signal_1 "signal comment";
CM_ SG_ 160 signal_2 "multiple line comment 
1
2
";

CM_ BO_ 162 "message comment with \"escaped quotes\"";
CM_ SG_ 162 signal_1 "signal comment with \"escaped quotes\"";
)";

  DBCFile file("", content);
  auto msg = file.msg(160);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->name == "message_1");
  REQUIRE(msg->size == 8);
  REQUIRE(msg->comment == "message comment");
  REQUIRE(msg->sigs.size() == 2);
  REQUIRE(msg->transmitter == "EON");
  REQUIRE(file.msg("message_1") != nullptr);

  auto sig_1 = msg->sigs[0];
  REQUIRE(sig_1->name == "signal_1");
  REQUIRE(sig_1->start_bit == 0);
  REQUIRE(sig_1->size == 12);
  REQUIRE(sig_1->min == 0);
  REQUIRE(sig_1->max == 4095);
  REQUIRE(sig_1->unit == "unit");
  REQUIRE(sig_1->comment == "signal comment");
  REQUIRE(sig_1->receiver_name == "XXX");
  REQUIRE(sig_1->val_desc.size() == 3);
  REQUIRE(sig_1->val_desc[0] == std::pair<double, std::string>{0, "disabled"});
  REQUIRE(sig_1->val_desc[1] == std::pair<double, std::string>{1.2, "initializing"});
  REQUIRE(sig_1->val_desc[2] == std::pair<double, std::string>{2, "fault"});

  auto &sig_2 = msg->sigs[1];
  REQUIRE(sig_2->comment == "multiple line comment \n1\n2");

  // multiplexed signals
  msg = file.msg(162);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->sigs.size() == 2);
  REQUIRE(msg->sigs[0]->type == cabana::Signal::Type::Multiplexor);
  REQUIRE(msg->sigs[1]->type == cabana::Signal::Type::Multiplexed);
  REQUIRE(msg->sigs[1]->multiplex_value == 4);
  REQUIRE(msg->sigs[1]->start_bit == 12);
  REQUIRE(msg->sigs[1]->size == 1);
  REQUIRE(msg->sigs[1]->receiver_name == "XXX");

  // escaped quotes
  REQUIRE(msg->comment == "message comment with \"escaped quotes\"");
  REQUIRE(msg->sigs[0]->comment == "signal comment with \"escaped quotes\"");
}

void test_parse_opendbc() {
  std::vector<std::string> errors;
  for (const auto &entry : std::filesystem::directory_iterator(OPENDBC_FILE_PATH)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".dbc") continue;
    try {
      auto dbc = DBCFile(entry.path().string());
    } catch (std::exception &e) {
      errors.push_back(e.what());
    }
  }
  std::ostringstream details;
  for (const auto &error : errors) details << error << '\n';
  if (!errors.empty()) std::cerr << details.str();
  REQUIRE(errors.empty());
}

void test_dbc_manager() {
  DBCManager manager;
  int files_changed = 0;
  int signals_added = 0;
  int masks_updated = 0;
  Connections connections;
  connections.push_back(manager.signalAdded.connect([&](MessageId, const cabana::Signal *) { ++signals_added; }));
  connections.push_back(manager.fileChanged.connect([&]() { ++files_changed; }));
  connections.push_back(manager.maskUpdated.connect([&]() { ++masks_updated; }));

  std::string error;
  REQUIRE(manager.open(SOURCE_ALL, "test", "BO_ 160 message: 8 XXX\n", &error));
  REQUIRE(error.empty());
  REQUIRE(files_changed == 1);

  cabana::Signal signal{};
  signal.name = "speed";
  signal.start_bit = 0;
  signal.size = 8;
  signal.is_little_endian = true;
  manager.addSignal({.source = 0, .address = 160}, signal);
  REQUIRE(signals_added == 1);
  REQUIRE(masks_updated == 1);
  REQUIRE(manager.msg({.source = 0, .address = 160})->sig("speed") != nullptr);
}

void test_format_seconds() {
  REQUIRE(utils::formatSeconds(0) == "00:00");
  REQUIRE(utils::formatSeconds(59.4) == "00:59");
  REQUIRE(utils::formatSeconds(-1) == "00:00");
  REQUIRE(utils::formatSeconds(61.234, true) == "01:01.23");
  REQUIRE(utils::formatSeconds(3599.9) == "59:59");
  REQUIRE(utils::formatSeconds(3601) == "01:00:01");
  REQUIRE(utils::formatSeconds(3601.5, true) == "01:00:01.50");

  const char *tz = getenv("TZ");
  const bool had_tz = tz != nullptr;
  const std::string saved_tz = had_tz ? tz : "";
  setenv("TZ", "UTC", 1);
  tzset();
  REQUIRE(utils::formatSeconds(0, false, true) == "1970-01-01 00:00:00");
  REQUIRE(utils::formatSeconds(1700000000.123, true, true) == "2023-11-14 22:13:20.12");
  if (had_tz) {
    setenv("TZ", saved_tz.c_str(), 1);
  } else {
    unsetenv("TZ");
  }
  tzset();
}

void test_to_hex() {
  REQUIRE(utils::toHex({}) == "");
  REQUIRE(utils::toHex({0x00, 0x0f, 0xab, 0xff}) == "000FABFF");
  REQUIRE(utils::toHex({0x01, 0x02, 0x03}, ' ') == "01 02 03");

  REQUIRE(utils::toHexString(0) == "0x00");
  REQUIRE(utils::toHexString(0xf) == "0x0F");
  REQUIRE(utils::toHexString(0x1ab) == "0x1AB");
  REQUIRE(utils::toHexString(0x1fffffff) == "0x1FFFFFFF");
}

void test_signal_tooltip() {
  cabana::Signal sig{};
  sig.name = "speed";
  sig.start_bit = 3;
  sig.size = 12;
  sig.msb = 14;
  sig.lsb = 3;
  sig.is_little_endian = true;
  sig.is_signed = false;
  REQUIRE(utils::signalToolTip(&sig) == R"(
    speed<br /><span font-size:small">
    Start Bit: 3 Size: 12<br />
    MSB: 14 LSB: 3<br />
    Little Endian: Y Signed: N</span>
  )");
}

void test_route_timestamps() {
  REQUIRE(routes::parseIsoToUnixMs("2024-01-02T03:04:05Z") == 1704164645000);
  REQUIRE(routes::parseIsoToUnixMs("2024-01-02T03:04:05") == 1704164645000);
  REQUIRE(routes::parseIsoToUnixMs("2024-01-02 03:04:05") == 1704164645000);
  REQUIRE(routes::parseIsoToUnixMs("2024-01-02T03:04:05.123Z") == 1704164645123);
  REQUIRE(routes::parseIsoToUnixMs("2024-01-02T03:04:05.4Z") == 1704164645400);
  REQUIRE(routes::parseIsoToUnixMs("2024-01-02T03:04:05.123456Z") == 1704164645123);
  REQUIRE(routes::parseIsoToUnixMs("") == 0);
  REQUIRE(routes::parseIsoToUnixMs("not a timestamp") == 0);

  // formatUnixMs is local time
  const char *tz = getenv("TZ");
  const std::string prev_tz = tz ? tz : "";
  setenv("TZ", "UTC", 1);
  tzset();
  REQUIRE(routes::formatUnixMs(1704164645123) == "2024-01-02 03:04:05");
  if (tz) {
    setenv("TZ", prev_tz.c_str(), 1);
  } else {
    unsetenv("TZ");
  }
  tzset();
}

void test_route_api_response() {
  REQUIRE(routes::checkApiResponse("") == std::make_pair(false, 500));
  REQUIRE(routes::checkApiResponse("not json") == std::make_pair(false, 500));
  REQUIRE(routes::checkApiResponse(R"({"error": "unauthorized"})") == std::make_pair(false, 401));
  REQUIRE(routes::checkApiResponse(R"({"error": "server error"})") == std::make_pair(false, 500));
  REQUIRE(routes::checkApiResponse("[]") == std::make_pair(true, 0));
  REQUIRE(routes::checkApiResponse(R"({"dongle_id": "aaaa"})") == std::make_pair(true, 0));
}

void test_route_json() {
  auto devices = routes::parseDevices(R"([{"dongle_id": "aaaa"}, {"dongle_id": "bbbb"}])");
  REQUIRE(devices.size() == 2);
  REQUIRE(devices[0].dongle_id == "aaaa");
  REQUIRE(devices[1].dongle_id == "bbbb");
  REQUIRE(routes::parseDevices("not json").empty());
  REQUIRE(routes::parseDevices(R"({"error": "unauthorized"})").empty());

  auto list = routes::parseRoutes(
      R"([{"fullname": "aaaa|2024-01-02--03-04-05", "start_time_utc_millis": 1704164645000, "end_time_utc_millis": 1704165245000}])", false);
  REQUIRE(list.size() == 1);
  REQUIRE(list[0].name == "aaaa|2024-01-02--03-04-05");
  REQUIRE(list[0].start_ms == 1704164645000);
  REQUIRE(list[0].end_ms == 1704165245000);

  // preserved routes report ISO-8601 timestamps
  auto preserved = routes::parseRoutes(
      R"([{"fullname": "aaaa|2024-01-02--03-04-05", "start_time": "2024-01-02T03:04:05Z", "end_time": "2024-01-02T03:14:05Z"}])", true);
  REQUIRE(preserved.size() == 1);
  REQUIRE(preserved[0].start_ms == 1704164645000);
  REQUIRE(preserved[0].end_ms == 1704165245000);

  REQUIRE(routes::parseRoutes("not json", false).empty());
}

static std::vector<uint8_t> fromHex(const std::string &hex) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back((uint8_t)std::stoul(hex.substr(i, 2), nullptr, 16));
  }
  return out;
}

void test_qt_state_blobs() {
  // blobs written by the Qt frontend
  auto geometry = qtstate::parseQtGeometry(fromHex(
      "01d9d0cb000300000000000000000014000004ff000003330000000000000014000004ff"
      "00000333000000000000000006400000000000000014000004ff00000333"));
  REQUIRE(geometry.has_value());
  REQUIRE(geometry->x == 0);
  REQUIRE(geometry->y == 20);
  REQUIRE(geometry->w == 1280);
  REQUIRE(geometry->h == 800);
  REQUIRE(geometry->maximized == false);

  auto splitter = qtstate::parseQtSplitter(fromHex("000000ff0000000100000002000000960000006801ffffffff010000000200"));
  REQUIRE(splitter.has_value());
  REQUIRE(std::fabs(splitter->ratio - 150.0f / 254.0f) < 1e-6f);

  auto header = qtstate::parseQtHeaderState(fromHex(
      "000000ff000000000000000100000000000000000100000000000000000000000000000000000003360000000701"
      "01000100000000000000000000000068ffffffff0000008400000000000000070000006800000001000000000000"
      "00680000000100000000000000680000000100000000000000680000000100000000000000680000000100000000"
      "000000680000000100000000000000c60000000100000002000003e800000000c6"));
  REQUIRE(header.has_value());
  REQUIRE(header->sort_section == 0);
  REQUIRE(header->sort_order == 0);
  REQUIRE(header->sort_shown == true);
  const int expected_width[] = {104, 104, 104, 104, 104, 104, 198};
  for (int i = 0; i < qtstate::kMessageColumnCount; ++i) {
    REQUIRE(header->visual[i] == i);
    REQUIRE(header->width[i] == expected_width[i]);
    REQUIRE(header->hidden[i] == false);
  }

  // empty, truncated and wrong magic blobs are rejected
  REQUIRE(!qtstate::parseQtGeometry({}).has_value());
  REQUIRE(!qtstate::parseQtSplitter({}).has_value());
  REQUIRE(!qtstate::parseQtHeaderState({}).has_value());
  REQUIRE(!qtstate::parseQtGeometry(fromHex("01d9d0cb00030000000000000000")).has_value());
  REQUIRE(!qtstate::parseQtSplitter(fromHex("000000ff000000010000000200000096")).has_value());
  REQUIRE(!qtstate::parseQtHeaderState(fromHex("000000ff0000000000000001000000000000000001")).has_value());
  REQUIRE(!qtstate::parseQtGeometry(fromHex("deadbeef000300000000000000000014000004ff00000333")).has_value());
  REQUIRE(!qtstate::parseQtSplitter(fromHex("000000fe0000000100000002000000960000006801ffffffff010000000200")).has_value());
  REQUIRE(!qtstate::parseQtHeaderState(fromHex("000000fe00000000000000010000000000000000010000000000000000")).has_value());
}

void test_parallel_failure_joins_workers() {
  for (bool fail_on_caller : {true, false}) {
    std::atomic<int> finished = 0;
    const size_t chunks = std::clamp<size_t>(std::thread::hardware_concurrency(), 2, 4) + 1;
    bool caught = false;
    try {
      parallelFor(chunks, [&](size_t begin, size_t end) {
        if (begin == (fail_on_caller ? 0u : 1u)) throw std::runtime_error("task failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ++finished;
      });
    } catch (const std::runtime_error &) {
      caught = true;
    }
    REQUIRE(caught);
    REQUIRE(finished == chunks - 1);
  }
}

void test_pixel_envelope() {
  struct Point {
    double x, y;
    Point(double x, double y) : x(x), y(y) {}
  };
  std::vector<Point> points;
  for (int i = 0; i < 1000; ++i) points.emplace_back(i * 0.001, i == 203 ? 99 : i == 201 ? -99 : 0);
  const auto result = chart::pixelEnvelope(points.begin(), points.end(), 0, 1, 10);
  REQUIRE(result.size() <= 40);
  REQUIRE(result.front().x == points.front().x);
  REQUIRE(result.back().x == points.back().x);
  REQUIRE(std::is_sorted(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.x < b.x; }));
  REQUIRE(std::any_of(result.begin(), result.end(), [](const auto &p) { return p.x == .201 && p.y == -99; }));
  REQUIRE(std::any_of(result.begin(), result.end(), [](const auto &p) { return p.x == .203 && p.y == 99; }));
  const std::vector<Point> step{{-1, 0}, {0, 0}, {0, 10}, {0, -10}, {0, 0}, {2, 0}};
  const auto edge = chart::pixelEnvelope(step.begin(), step.end(), 0, 1, 2);
  REQUIRE(edge.front().x == -1);
  REQUIRE(edge.back().x == 2);
  REQUIRE(std::any_of(edge.begin(), edge.end(), [](const auto &p) { return p.y == 10; }));
  REQUIRE(std::any_of(edge.begin(), edge.end(), [](const auto &p) { return p.y == -10; }));
  REQUIRE(chart::pixelEnvelope(points.begin(), points.begin(), 0, 1, 10).empty());
}

void test_chart_analysis() {
  struct Point { double x, y; Point(double x, double y) : x(x), y(y) {} };
  const std::vector<Point> raw{{0, 2}, {1, 4}, {3, 8}, {3, 10}, {4, 12}};
  auto original = chart::transform(raw, {});
  REQUIRE(original.size() == raw.size());
  REQUIRE(original.front().x == 0);
  REQUIRE(original.front().y == 2);
  auto scaled = chart::transform(raw, {chart::Transform::None, -2, 1});
  REQUIRE(scaled.back().y == -23);
  auto derivative = chart::transform(raw, {chart::Transform::Derivative});
  REQUIRE(derivative.size() == 3);  // omit first point and duplicate timestamp
  REQUIRE(derivative[0].x == 1);
  REQUIRE(derivative[0].y == 2);
  REQUIRE(derivative[1].x == 3);
  REQUIRE(derivative[1].y == 2);
  REQUIRE(derivative[2].y == 2);
  auto integral = chart::transform(raw, {chart::Transform::Integral});
  REQUIRE(integral.front().y == 0);
  REQUIRE(integral[2].y == 15);  // trapezoids over unequal time steps
  REQUIRE(integral[3].y == 15);
  REQUIRE(integral.back().y == 26);
  auto average = chart::transform(raw, {chart::Transform::MovingAverage, 2, 1, 2});
  REQUIRE(average[0].y == 5);
  REQUIRE(average[1].y == 7);
  REQUIRE(average[2].y == 13);
  REQUIRE(average.back().y == 23);
  // A streaming processor produces the same values when a batch boundary falls between samples.
  for (auto type : {chart::Transform::None, chart::Transform::Derivative, chart::Transform::Integral, chart::Transform::MovingAverage}) {
    chart::TransformSettings settings{type, -2, 3, 3};
    const auto expected = chart::transform(raw, settings);
    chart::TransformState state;
    std::vector<Point> streamed;
    for (size_t batch = 0; batch < raw.size(); batch += 2) {
      for (size_t i = batch; i < std::min(batch + 2, raw.size()); ++i) {
        if (auto value = state.append(raw[i].x, raw[i].y, settings)) streamed.emplace_back(raw[i].x, *value);
      }
    }
    REQUIRE(streamed.size() == expected.size());
    for (size_t i = 0; i < streamed.size(); ++i) {
      REQUIRE(streamed[i].x == expected[i].x);
      REQUIRE(streamed[i].y == expected[i].y);
    }
  }
  REQUIRE(chart::transform(std::vector<Point>{}, {}).empty());
  REQUIRE(chart::transform(std::vector<Point>{{0, 0}}, {chart::Transform::Derivative}).empty());
  REQUIRE(chart::transform(std::vector<Point>{{0, 0}}, {}).front().y == 0);
  REQUIRE(chart::csvField("signal, \"left\"\n") == "\"signal, \"\"left\"\"\n\"");
}

void test_pixel_envelope() {
  using Point = cabana::Sample;
  std::vector<Point> points;
  for (int i = 0; i < 1000; ++i) points.emplace_back(i * 0.001, i == 203 ? 99 : i == 201 ? -99 : 0);
  const auto result = chart::pixelEnvelope(points.begin(), points.end(), 0, 1, 10);
  REQUIRE(result.size() <= 40);
  REQUIRE(result.front().x == points.front().x);
  REQUIRE(result.back().x == points.back().x);
  REQUIRE(std::is_sorted(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.x < b.x; }));
  REQUIRE(std::any_of(result.begin(), result.end(), [](const auto &p) { return p.x == .201 && p.y == -99; }));
  REQUIRE(std::any_of(result.begin(), result.end(), [](const auto &p) { return p.x == .203 && p.y == 99; }));
  const std::vector<Point> step{{-1, 0}, {0, 0}, {0, 10}, {0, -10}, {0, 0}, {2, 0}};
  const auto edge = chart::pixelEnvelope(step.begin(), step.end(), 0, 1, 2);
  REQUIRE(edge.front().x == -1);
  REQUIRE(edge.back().x == 2);
  REQUIRE(std::any_of(edge.begin(), edge.end(), [](const auto &p) { return p.y == 10; }));
  REQUIRE(std::any_of(edge.begin(), edge.end(), [](const auto &p) { return p.y == -10; }));
  REQUIRE(chart::pixelEnvelope(points.begin(), points.begin(), 0, 1, 10).empty());
}

void test_chart_layout() {
  using json11::Json;
  Json::object signal{{"message", "2:1AF"}, {"signal", "Speed"}, {"visible", false}, {"transform", 3},
                      {"scale", -2.5}, {"offset", 1.0}, {"window", 20}};
  auto document = [&](const Json &s) {
    return Json(Json::object{{"cabana_layout", 1}, {"columns", 2}, {"range", 60},
      {"tabs", Json::array{Json::array{Json::object{{"type", 1}, {"signals", Json::array{s}}}}, Json::array{}}}}).dump();
  };
  auto layout = chart::parseLayout(document(signal));
  REQUIRE(layout.has_value());
  REQUIRE(layout->tabs.size() == 2);
  REQUIRE(layout->tabs[1].empty());
  const auto &s = layout->tabs[0][0].signals[0];
  REQUIRE(s.id.source == 2);
  REQUIRE(s.id.address == 0x1af);
  REQUIRE(!s.visible);
  REQUIRE(s.transform.scale == -2.5);
  REQUIRE(s.transform.window == 20);
  for (const auto &bad_id : {"bad", "x:1", "256:1", "1:100000000", "0:", ":1", "0:1junk", "-1:1"}) {
    auto bad = signal;
    bad["message"] = bad_id;
    REQUIRE(!chart::parseLayout(document(bad)).has_value());
  }
  for (const auto &key : {"visible", "transform", "window", "scale", "signal"}) {
    auto bad = signal;
    bad.erase(key);
    REQUIRE(!chart::parseLayout(document(bad)).has_value());
  }
  auto bad = signal;
  bad["window"] = 0;
  REQUIRE(!chart::parseLayout(document(bad)).has_value());
  bad["window"] = 1.5;
  REQUIRE(!chart::parseLayout(document(bad)).has_value());
  REQUIRE(!chart::parseLayout("{}").has_value());
  REQUIRE(!chart::parseLayout("{truncated").has_value());
}

void test_cereal_telemetry() {
  capnp::MallocMessageBuilder message;
  auto event = message.initRoot<cereal::Event>();
  event.setLogMonoTime(1000000000);
  event.setValid(true);
  auto state = event.initCarState();
  state.setVEgo(12.5);
  state.setAEgo(0);
  state.setSteeringPressed(false);
  state.setGearShifter(cereal::CarState::GearShifter::DRIVE);
  cabana::Telemetry data;
  cabana::extractTelemetry(event.asReader(), data);
  REQUIRE(data.at("/carState/vEgo").front().y == 12.5);
  REQUIRE(data.at("/carState/aEgo").front().y == 0);
  REQUIRE(data.at("/carState/steeringPressed").front().y == 0);
  REQUIRE(data.at("/carState/gearShifter").front().y == (int)cereal::CarState::GearShifter::DRIVE);
  REQUIRE(data.at("/carState/__logMonoTimeSeconds").front().y == 1);
  REQUIRE(data.at("/carState/__valid").front().y == 1);
  auto control = event.initCarControl();
  auto orientation = control.initOrientationNED(3);
  orientation.set(0, 0.125);
  orientation.set(1, 0);
  orientation.set(2, -1.5);
  cabana::extractTelemetry(event.asReader(), data);
  REQUIRE(data.at("/carControl/orientationNED/0").front().y == 0.125);
  REQUIRE(data.at("/carControl/orientationNED/1").front().y == 0);
  REQUIRE(data.at("/carControl/orientationNED/2").front().y == -1.5);
  cabana::Telemetry earlier{{"/carState/vEgo", {{0.5, 10}}}};
  cabana::mergeTelemetry(data, std::move(earlier));
  REQUIRE(data.at("/carState/vEgo").front().x == 0.5);
  REQUIRE(data.at("/carState/vEgo").back().x == 1);
}

void require_same_telemetry(const cabana::Telemetry &expected, const cabana::Telemetry &actual) {
  REQUIRE(actual.size() == expected.size());
  for (const auto &[path, samples] : expected) {
    const auto &other = actual.at(path);
    REQUIRE(samples.size() == other.size());
    for (size_t i = 0; i < samples.size(); ++i) {
      REQUIRE(samples[i].x == other[i].x);
      REQUIRE(samples[i].y == other[i].y);
    }
  }
}

void test_cached_telemetry_extractor() {
  cabana::Telemetry expected, actual;
  cabana::TelemetryExtractor extractor(actual);
  for (int i = 0; i < 100; ++i) {
    capnp::MallocMessageBuilder message;
    auto event = message.initRoot<cereal::Event>();
    event.setLogMonoTime(1000000000ULL + i * 10000000ULL);
    event.setValid(i % 3 != 0);
    switch (i % 5) {
      case 0: {
        auto state = event.initCarState();
        state.setVEgo(i % 2 ? std::numeric_limits<float>::quiet_NaN() : i);
        state.setSteeringPressed(i % 2);
        break;
      }
      case 1: {
        auto values = event.initCarControl().initOrientationNED(i % 7);
        for (size_t j = 0; j < values.size(); ++j) values.set(j, j == 2 ? std::numeric_limits<float>::infinity() : i - j * 0.25);
        break;
      }
      case 2: {
        auto sensor = event.initAccelerometer();
        if (i % 2) sensor.setTemperature(i);
        else {
          auto values = sensor.initAcceleration().initV(i % 4);
          for (size_t j = 0; j < values.size(); ++j) values.set(j, i + j * 0.125);
        }
        break;
      }
      case 3: {
        auto lateral = event.initControlsState().initLateralControlState();
        if (i % 2) lateral.initPidState().setActive(true);
        else lateral.initAngleState().setActive(false);
        break;
      }
      case 4:
        if (i % 2) event.initCan(1);
        else event.initSendcan(1);
        break;
    }
    cabana::extractTelemetry(event.asReader(), expected);
    extractor.extract(event.asReader());
  }
  require_same_telemetry(expected, actual);
}

void test_prepared_telemetry_merge() {
  const cabana::Telemetry published{{"a", {{2, 20}, {4, 40}}}, {"unchanged", {{1, 10}}}};
  for (const auto &samples : std::vector<std::vector<cabana::Sample>>{
         {}, {{0, 0}}, {{5, 50}}, {{1, 10}, {2, 21}, {3, 30}, {6, 60}}}) {
    cabana::Telemetry batch{{"a", samples}, {"new", {{1, 100}}}};
    auto expected = published;
    cabana::mergeTelemetry(expected, batch);
    cabana::TelemetrySnapshot snapshot;
    for (const auto &[path, points] : published) snapshot[path] = std::make_shared<const cabana::Samples>(points);
    cabana::prepareTelemetryMerge(snapshot, batch);
    REQUIRE(!batch.count("unchanged"));
    REQUIRE(published.at("a").size() == 2);
    REQUIRE(published.at("a").front().y == 20);
    auto actual = published;
    for (auto &[path, points] : batch) actual[path].swap(points);
    REQUIRE(actual.size() == expected.size());
    for (const auto &[path, points] : expected) {
      REQUIRE(actual.at(path).size() == points.size());
      for (size_t i = 0; i < points.size(); ++i) {
        REQUIRE(actual.at(path)[i].x == points[i].x);
        REQUIRE(actual.at(path)[i].y == points[i].y);
      }
    }
    REQUIRE(batch.at("a").size() == 2);  // retired data is owned by the worker's batch
  }
}

void test_layout_equations() {
  cabana::Telemetry data{{"speed", {{0, 10}, {1, 20}, {2, 30}}}, {"enabled", {{0, 0}, {1.5, 1}}}};
  REQUIRE(cabana::nearestValue(data.at("enabled"), 0.75) == 1);  // tie: later sample, as PlotJuggler
  REQUIRE(cabana::nearestValue(data.at("enabled"), -1) == 0);
  REQUIRE(cabana::nearestValue(data.at("enabled"), 5) == 1);
  cabana::Equation equation{"scaled", "speed", "sum = 0", "global sum\nsum += value\nreturn sum * v1", {"enabled"}};
  auto values = cabana::evaluateEquation(equation, data);
  REQUIRE(values.size() == 3);
  REQUIRE(values[0].y == 0);
  REQUIRE(values[1].y == 30);
  REQUIRE(values[2].y == 60);
  REQUIRE(cabana::evaluateEquation(equation, data)[2].y == 60);  // state resets when reloading earlier data
  equation.function = "return time + 1, abs(value)";
  REQUIRE(cabana::evaluateEquation(equation, data)[0].x == 1);
  equation.globals = "import statistics";
  equation.function = "return statistics.mean((value, v1))";
  REQUIRE(cabana::evaluateEquation(equation, data)[0].y == 5);
  equation.globals.clear();
  for (auto code : {"raise ValueError('bad equation')", "while True:\n  pass", "invalid Python !", "return None", "return (1, 2, 3)"}) {
    equation.function = code;
    bool failed = false;
    try { cabana::evaluateEquation(equation, data); } catch (const std::exception &) { failed = true; }
    REQUIRE(failed);
  }
}

void test_signal_tree() {
  chart::SignalTree tree;
  tree.rebuild({"/carState/vEgo", "/carState/aEgo", "/model/accel/10", "/model/accel/2", "/model/accel/0", "speed error"});
  tree.filter("");
  REQUIRE(tree.nodes[0].matches == 6);
  auto rows = tree.visible({});
  REQUIRE(rows.size() == 3);
  REQUIRE(tree.nodes[rows[0]].name == "carState");
  REQUIRE(tree.nodes[rows[1]].name == "model");
  REQUIRE(tree.nodes[rows[2]].path == "speed error");
  rows = tree.visible({"/model", "/model/accel"});
  REQUIRE(rows.size() == 7);
  REQUIRE(tree.nodes[rows[3]].name == "0");
  REQUIRE(tree.nodes[rows[4]].name == "2");
  REQUIRE(tree.nodes[rows[5]].name == "10");
  REQUIRE(tree.nodes[rows[5]].path == "/model/accel/10");
  tree.filter("VEGO");
  REQUIRE(tree.nodes[0].matches == 1);
  rows = tree.visible({"/carState"});
  REQUIRE(rows.size() == 2);
  REQUIRE(tree.nodes[rows[1]].path == "/carState/vEgo");
  tree.filter("model/accel");
  REQUIRE(tree.nodes[0].matches == 3);
  tree.filter("missing");
  REQUIRE(tree.visible({}).empty());
  tree.rebuild({"/carState/vEgo", "/carState/vEgo"});
  tree.filter("");
  REQUIRE(tree.nodes[0].matches == 1);
  tree.rebuild({});
  tree.filter("");
  REQUIRE(tree.visible({}).empty());
}

void test_cabana_core() {
  test_pixel_envelope();
  test_signal_tree();
  test_cereal_telemetry();
  test_cached_telemetry_extractor();
  test_prepared_telemetry_merge();
  test_layout_equations();
  test_chart_analysis();
  test_pixel_envelope();
  test_chart_layout();
  test_format_seconds();
  test_to_hex();
  test_message_id_parsing();
  test_signal_tooltip();
  test_generate_dbc();
  test_comment_order();
  test_preserve_original_header();
  test_escaped_quotes();
  test_parse_dbc();
  test_parse_opendbc();
  test_dbc_manager();
  test_route_timestamps();
  test_route_api_response();
  test_route_json();
  test_parallel_failure_joins_workers();
  test_qt_state_blobs();
}

int main(int argc, char **argv) {
  if (argc == 3 && std::string(argv[1]) == "--check-downloader") {
    return run_native_test([&]() {
      const std::string mode = argv[2];
      const std::string prefix = std::getenv("OPENPILOT_PREFIX");
      bool progress = false;
      installDownloadProgressHandler([&](uint64_t current, uint64_t total, bool success) {
        if (success && current == 42 && total == 100) progress = true;
      });
      std::atomic<bool> abort = false;
      std::thread cancel;
      if (mode == "abort") cancel = std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        abort = true;
      });
      const auto result = PyDownloader::download(mode == "ok" ? "url with spaces & literal $value" : mode, true, &abort);
      if (cancel.joinable()) cancel.join();
      installDownloadProgressHandler(nullptr);
      REQUIRE(std::string(std::getenv("OPENPILOT_PREFIX")) == prefix);
      if (mode == "ok") {
        REQUIRE(result == "downloaded path");
        REQUIRE(progress);
      } else {
        REQUIRE(result.empty());
      }
    });
  }
  if (argc == 3 && std::string(argv[1]) == "--check-telemetry") {
    return run_native_test([&]() {
      LogReader log;
      REQUIRE(log.load(argv[2]));
      cabana::Telemetry expected, actual;
      cabana::TelemetryExtractor extractor(actual);
      for (const auto &event : log.events) {
        capnp::FlatArrayMessageReader reader(event.data);
        cabana::extractTelemetry(reader.getRoot<cereal::Event>(), expected);
        extractor.extract(reader.getRoot<cereal::Event>());
      }
      require_same_telemetry(expected, actual);
      require_same_telemetry(expected, cabana::extractLogTelemetry(log, std::atomic<bool>{false}));
      REQUIRE(cabana::extractLogTelemetry(log, std::atomic<bool>{true}).empty());
      size_t count = 0;
      for (const auto &[path, samples] : actual) count += samples.size();
      printf("Verified %zu paths and %zu samples across %zu events\n", actual.size(), count, log.events.size());
    });
  }
  if (argc == 3 && std::string(argv[1]) == "--check-layout") {
    return run_native_test([&]() {
      std::ifstream in(argv[2]);
      const std::string contents{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
      auto layout = chart::parseLayout(contents);
      REQUIRE(layout.has_value());
      for (const auto &e : layout->equations) {
        cabana::Telemetry data;
        auto add = [&](const std::string &path) { for (int i = 0; i < 10; ++i) data[path].emplace_back(100 + i, 1); };
        add(e.source);
        for (const auto &path : e.additional) add(path);
        auto values = cabana::evaluateEquation(e, data);
        REQUIRE(values.size() == 10);
        for (const auto &value : values) REQUIRE(std::isfinite(value.y));
        if (e.name == "engaged curvature yaw") {
          for (int i = 0; i < 10; ++i) {
            data[e.source][i].y = 0.02;
            data["/carState/vEgo"][i].y = 20;
            data["/carState/steeringPressed"][i].y = i < 2 ? 1 : 0;
          }
          values = cabana::evaluateEquation(e, data);
          REQUIRE(values.size() == 10);
          for (int i = 0; i <= 6; ++i) REQUIRE(values[i].y == 0);
          for (int i = 7; i < 10; ++i) REQUIRE(std::abs(values[i].y - 0.001) < 1e-12);
        }
      }
    });
  }
  return run_native_test(test_cabana_core);
}
