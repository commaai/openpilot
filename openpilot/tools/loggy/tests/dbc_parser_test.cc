#include <filesystem>
#include <string>

#include "catch2/catch.hpp"
#include "tools/loggy/backend/dbc/dbcfile.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"

namespace fs = std::filesystem;

namespace {

fs::path opendbcPath() {
#ifdef OPENDBC_FILE_PATH
  fs::path configured(OPENDBC_FILE_PATH);
  if (fs::exists(configured)) return configured;
#endif

#ifdef LOGGY_REPO_ROOT
  return fs::path(LOGGY_REPO_ROOT) / "opendbc_repo" / "opendbc" / "dbc";
#else
  return fs::path("opendbc_repo") / "opendbc" / "dbc";
#endif
}

}  // namespace

TEST_CASE("DBCFile parses inline DBC content") {
  const std::string content = R"(
BO_ 160 message_1: 8 EON
  SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit"  XXX
  SG_ signal_2 : 12|1@1+ (1.0,0.0) [0.0|1] ""  XXX

BO_ 162 message_2: 8 XXX
  SG_ mux M : 0|4@1+ (1,0) [0|15] "" XXX
  SG_ gated M4 : 12|1@1+ (1.0,0.0) [0.0|1] "" XXX

VAL_ 160 signal_1 0 "disabled" 1.2 "initializing" 2 "fault";

CM_ BO_ 160 "message comment";
CM_ SG_ 160 signal_1 "signal comment";
)";

  loggy::DBCFile file("", content);
  auto *msg = file.msg(160);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->name == "message_1");
  REQUIRE(msg->size == 8);
  REQUIRE(msg->comment == "message comment");
  REQUIRE(msg->sigs.size() == 2);

  auto *sig = msg->sig("signal_1");
  REQUIRE(sig != nullptr);
  REQUIRE(sig->unit == "unit");
  REQUIRE(sig->comment == "signal comment");
  REQUIRE(sig->val_desc.size() == 3);
  REQUIRE(sig->val_desc[1] == std::pair<double, std::string>{1.2, "initializing"});

  auto *mux_msg = file.msg(162);
  REQUIRE(mux_msg != nullptr);
  REQUIRE(mux_msg->sigs.size() == 2);
  REQUIRE(mux_msg->sigs[0]->type == loggy::Signal::Type::Multiplexor);
  REQUIRE(mux_msg->sigs[1]->type == loggy::Signal::Type::Multiplexed);
  REQUIRE(mux_msg->sigs[1]->multiplex_value == 4);
}

TEST_CASE("DBCFile writer preserves documented legacy omissions") {
  const std::string content = R"(VERSION ""

BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

BO_ 161 signal_less: 8 EON

BA_ "GenMsgCycleTime" BO_ 160 10;
BO_TX_BU_ 160 : EON;
)";

  loggy::DBCFile dbc("", content);
  const std::string generated = dbc.generateDBC();
  CHECK(generated.find("BO_ 160 message_1") != std::string::npos);
  CHECK(generated.find("BA_ ") == std::string::npos);
  CHECK(generated.find("BO_TX_BU_") == std::string::npos);
  CHECK(generated.find("BO_ 161 signal_less") == std::string::npos);
}

TEST_CASE("DBCFile parses ford_lincoln_base_pt.dbc regression") {
  const fs::path ford_dbc = opendbcPath() / "ford_lincoln_base_pt.dbc";
  REQUIRE(fs::exists(ford_dbc));

  loggy::DBCFile dbc(ford_dbc.string());
  REQUIRE(dbc.getMessages().size() > 100);

  auto *dte = dbc.msg(823);
  REQUIRE(dte != nullptr);
  CHECK(dte->name == "DTE_HPCMtoECG");
  CHECK(dte->sig("DteAcceptNew_B_Rq") != nullptr);
  CHECK(dte->sig("DteAcceptNew_B_Rq")->val_desc.size() == 2);
}

TEST_CASE("DBCManager source parsing and source-specific close fallback") {
  loggy::SourceSet sources;
  std::string error;
  REQUIRE(loggy::parseSourceSet("all", &sources, &error));
  CHECK(sources == loggy::SOURCE_ALL);
  CHECK(error.empty());
  REQUIRE(loggy::parseSourceSet("0, 2 3", &sources, &error));
  const loggy::SourceSet expected_sources{0, 2, 3};
  CHECK(sources == expected_sources);
  REQUIRE_FALSE(loggy::parseSourceSet("bad", &sources, &error));
  CHECK(error == "invalid source: bad");
  REQUIRE_FALSE(loggy::parseSourceSet("999", &sources, &error));
  CHECK(error == "invalid source: 999");

  loggy::DBCManager manager;
  REQUIRE(manager.open(loggy::SOURCE_ALL, "all", R"(
VERSION ""
BO_ 291 ALL_MSG: 1 XXX
 SG_ all_sig : 0|8@1+ (1,0) [0|255] "" XXX
)", &error));
  REQUIRE(manager.open(loggy::SourceSet{0}, "bus0", R"(
VERSION ""
BO_ 291 BUS0_MSG: 1 XXX
 SG_ bus0_sig : 0|8@1+ (1,0) [0|255] "" XXX
)", &error));

  REQUIRE(manager.findDBCFile(0) != nullptr);
  CHECK(manager.findDBCFile(0)->name() == "bus0");
  REQUIRE(manager.findDBCFile(1) != nullptr);
  CHECK(manager.findDBCFile(1)->name() == "all");

  manager.close(loggy::SourceSet{0});
  REQUIRE(manager.findDBCFile(0) != nullptr);
  CHECK(manager.findDBCFile(0)->name() == "all");
  CHECK(manager.allDBCFiles().size() == 1);
}

TEST_CASE("DBCFile saveAs preserves filename on write failure") {
  loggy::DBCFile dbc("inline", R"(
VERSION ""
BO_ 291 TEST_MSG: 1 XXX
 SG_ sig : 0|8@1+ (1,0) [0|255] "" XXX
)");
  dbc.filename = "/tmp/loggy-original.dbc";

  const fs::path bad_path = fs::temp_directory_path() / "loggy_missing_dir" / "out.dbc";
  if (fs::exists(bad_path.parent_path())) fs::remove_all(bad_path.parent_path());
  CHECK_FALSE(dbc.saveAs(bad_path.string()));
  CHECK(dbc.filename == "/tmp/loggy-original.dbc");
}
