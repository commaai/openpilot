
#undef INFO
#include <filesystem>
#include <sstream>

#include "catch2/catch.hpp"
#include "tools/cabana/dbc/dbcfile.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/core/settings.h"

#ifdef QT_CORE_LIB
#include <QColor>
#endif

const std::string TEST_RLOG_URL = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";

TEST_CASE("DBCFile::generateDBC") {
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

TEST_CASE("DBCFile::generateDBC - comment order") {
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

TEST_CASE("DBCFile::generateDBC -- preserve original header") {
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

TEST_CASE("DBCFile::generateDBC - escaped quotes") {
  std::string content = R"(BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment with \"escaped quotes\"";
CM_ SG_ 160 signal_1 "signal comment with \"escaped quotes\"";
)";
  DBCFile dbc("", content);
  REQUIRE(dbc.generateDBC() == content);
}

TEST_CASE("parse_dbc") {
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

TEST_CASE("parse_opendbc") {
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
  INFO(details.str());
  REQUIRE(errors.empty());
}

TEST_CASE("DBCManager core callbacks") {
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

TEST_CASE("Cabana settings core defaults") {
  CabanaSettingsState state;
  REQUIRE(state.fps == 10);
  REQUIRE(state.chart_range == 180);
  REQUIRE(state.drag_direction == CabanaSettingsState::MsbFirst);
  REQUIRE(state.recent_files.empty());
}

#ifdef QT_CORE_LIB
TEST_CASE("CabanaColor preserves QColor transformations") {
  const std::vector<QColor> colors = {
    QColor(102, 86, 169, 64), QColor(0, 187, 255, 128), QColor(255, 0, 0, 128), QColor(45, 120, 75, 255),
  };
  for (const auto &qt_color : colors) {
    CabanaColor color(qt_color.red(), qt_color.green(), qt_color.blue(), qt_color.alpha());
    for (int factor : {75, 100, 135, 150, 200}) {
      const auto lighter = color.lighter(factor);
      const auto qt_lighter = qt_color.lighter(factor);
      CHECK(std::abs(lighter.red() - qt_lighter.red()) <= 1);
      CHECK(std::abs(lighter.green() - qt_lighter.green()) <= 1);
      CHECK(std::abs(lighter.blue() - qt_lighter.blue()) <= 1);

      const auto darker = color.darker(factor);
      const auto qt_darker = qt_color.darker(factor);
      CHECK(std::abs(darker.red() - qt_darker.red()) <= 1);
      CHECK(std::abs(darker.green() - qt_darker.green()) <= 1);
      CHECK(std::abs(darker.blue() - qt_darker.blue()) <= 1);
    }
  }
}
#endif
