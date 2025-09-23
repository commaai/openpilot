
#undef INFO
#include <QDir>

#include "catch2/catch.hpp"
#include "tools/cabana/dbc/dbcmanager.h"

const std::string TEST_RLOG_URL = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";

TEST_CASE("DBCFile::generateDBC") {
  QString fn = QString("%1/%2.dbc").arg(OPENDBC_FILE_PATH, "tesla_can");
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
  auto content = R"(BO_ 160 message_1: 8 EON
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
  QString content = R"(VERSION "1.0"

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
  QString content = R"(BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment with \"escaped quotes\"";
CM_ SG_ 160 signal_1 "signal comment with \"escaped quotes\"";
)";
  DBCFile dbc("", content);
  REQUIRE(dbc.generateDBC() == content);
}

TEST_CASE("parse_dbc") {
  QString content = R"(
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
  REQUIRE(sig_1->val_desc[0] == std::pair<double, QString>{0, "disabled"});
  REQUIRE(sig_1->val_desc[1] == std::pair<double, QString>{1.2, "initializing"});
  REQUIRE(sig_1->val_desc[2] == std::pair<double, QString>{2, "fault"});

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
  QDir dir(OPENDBC_FILE_PATH);
  QStringList errors;
  for (auto fn : dir.entryList({"*.dbc"}, QDir::Files, QDir::Name)) {
    try {
      auto dbc = DBCFile(dir.filePath(fn));
    } catch (std::exception &e) {
      errors.push_back(e.what());
    }
  }
  INFO(errors.join("\n").toStdString());
  REQUIRE(errors.empty());
}
