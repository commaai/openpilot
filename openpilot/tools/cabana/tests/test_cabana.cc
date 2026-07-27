
#include <filesystem>
#include <sstream>

#include "common/tests/native_test.h"
#include "tools/cabana/dbc/dbcfile.h"
#include "tools/cabana/dbc/dbcmanager.h"

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
  manager.setCallbacks({
    .signal_added = [&](MessageId, const cabana::Signal *) { ++signals_added; },
    .file_changed = [&]() { ++files_changed; },
    .mask_updated = [&]() { ++masks_updated; },
  });

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

void test_cabana_core() {
  test_generate_dbc();
  test_comment_order();
  test_preserve_original_header();
  test_escaped_quotes();
  test_parse_dbc();
  test_parse_opendbc();
  test_dbc_manager();
}

int main() {
  return run_native_test(test_cabana_core);
}
