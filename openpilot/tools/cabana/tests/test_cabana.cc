
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <sstream>

#include "common/tests/native_test.h"
#include "tools/cabana/dbc/dbcfile.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/routes.h"
#include "tools/cabana/utils/strings.h"

const std::string TEST_RLOG_URL = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";

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
  REQUIRE(utils::formatSeconds(61.234, true) == "01:01.234");
  REQUIRE(utils::formatSeconds(3599.9) == "59:59");
  REQUIRE(utils::formatSeconds(3601) == "01:00:01");
  REQUIRE(utils::formatSeconds(3601.5, true) == "01:00:01.500");

  const char *tz = getenv("TZ");
  const bool had_tz = tz != nullptr;
  const std::string saved_tz = had_tz ? tz : "";
  setenv("TZ", "UTC", 1);
  tzset();
  REQUIRE(utils::formatSeconds(0, false, true) == "1970-01-01 00:00:00");
  REQUIRE(utils::formatSeconds(1700000000.123, true, true) == "2023-11-14 22:13:20.123");
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

void test_cabana_core() {
  test_format_seconds();
  test_to_hex();
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
}

int main() {
  return run_native_test(test_cabana_core);
}
