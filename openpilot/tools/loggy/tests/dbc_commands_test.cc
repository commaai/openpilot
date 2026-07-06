#include "catch2/catch.hpp"

#include "tools/loggy/backend/dbc/undo.h"

#include <string>
#include <vector>

TEST_CASE("DBC edit signal command applies undo and redo") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SourceSet{0}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ flag : 8|1@1+ (1,0) [0|1] "" XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  const loggy::Signal *origin = msg->sig("speed");
  REQUIRE(origin != nullptr);

  loggy::Signal edited = *origin;
  edited.name = "vehicle_speed";
  edited.factor = 0.5;
  edited.offset = 1.0;
  edited.min = -10.0;
  edited.max = 120.0;
  edited.unit = "m/s";
  edited.comment = "edited";
  edited.val_desc = {{4.0, "cruise"}};
  edited.precision = 3;
  edited.precision_override = true;
  edited.color = loggy::ColorRGBA{12, 34, 56, 255};
  edited.color_override = true;
  REQUIRE(loggy::commit_signal_edit(undo, manager, id, *origin, edited, error));
  CHECK(error.empty());
  CHECK(undo.count() == 1);
  CHECK(undo.can_undo());
  CHECK_FALSE(undo.can_redo());
  std::vector<loggy::UndoStackEntry> entries = undo.entries();
  REQUIRE(entries.size() == 1);
  CHECK(entries[0].index == 0);
  CHECK(entries[0].applied);
  CHECK_FALSE(entries[0].next);
  CHECK_FALSE(entries[0].clean);
  CHECK(entries[0].text.find("edit signal speed") != std::string::npos);
  undo.set_clean();
  entries = undo.entries();
  REQUIRE(entries.size() == 1);
  CHECK(entries[0].clean);
  REQUIRE(msg->sig("speed") == nullptr);
  REQUIRE(msg->sig("vehicle_speed") != nullptr);
  CHECK(msg->sig("vehicle_speed")->factor == 0.5);
  CHECK(msg->sig("vehicle_speed")->comment == "edited");
  CHECK(msg->sig("vehicle_speed")->val_desc == loggy::ValueDescription{{4.0, "cruise"}});
  CHECK(msg->sig("vehicle_speed")->format_value(3.0) == "cruise");
  CHECK(msg->sig("vehicle_speed")->precision == 3);
  CHECK(msg->sig("vehicle_speed")->precision_override);
  CHECK(msg->sig("vehicle_speed")->color.r == 12);
  CHECK(msg->sig("vehicle_speed")->color.g == 34);
  CHECK(msg->sig("vehicle_speed")->color.b == 56);
  CHECK(msg->sig("vehicle_speed")->color.a == 255);
  CHECK(msg->sig("vehicle_speed")->color_override);

  undo.undo();
  CHECK_FALSE(undo.can_undo());
  CHECK(undo.can_redo());
  entries = undo.entries();
  REQUIRE(entries.size() == 1);
  CHECK_FALSE(entries[0].applied);
  CHECK(entries[0].next);
  CHECK_FALSE(undo.is_clean());
  REQUIRE(msg->sig("speed") != nullptr);
  CHECK(msg->sig("vehicle_speed") == nullptr);
  CHECK(msg->sig("speed")->factor == 1.0);
  CHECK(msg->sig("speed")->val_desc.empty());
  CHECK_FALSE(msg->sig("speed")->precision_override);
  CHECK_FALSE(msg->sig("speed")->color_override);

  undo.redo();
  CHECK(undo.can_undo());
  entries = undo.entries();
  REQUIRE(entries.size() == 1);
  CHECK(entries[0].applied);
  CHECK_FALSE(entries[0].next);
  CHECK(undo.is_clean());
  REQUIRE(msg->sig("vehicle_speed") != nullptr);
  CHECK(msg->sig("vehicle_speed")->unit == "m/s");
  CHECK(msg->sig("vehicle_speed")->val_desc == loggy::ValueDescription{{4.0, "cruise"}});
  CHECK(msg->sig("vehicle_speed")->precision == 3);
  CHECK(msg->sig("vehicle_speed")->color.r == 12);
}

TEST_CASE("DBC edit signal command rejects duplicate names and missing targets") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SourceSet{0}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ flag : 8|1@1+ (1,0) [0|1] "" XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  const loggy::Signal *origin = msg->sig("speed");
  REQUIRE(origin != nullptr);

  loggy::Signal duplicate = *origin;
  duplicate.name = "flag";
  REQUIRE_FALSE(loggy::commit_signal_edit(undo, manager, id, *origin, duplicate, error));
  CHECK(error == "duplicate signal name: flag");
  CHECK(undo.count() == 0);

  loggy::Signal unnamed = *origin;
  unnamed.name.clear();
  REQUIRE_FALSE(loggy::commit_signal_edit(undo, manager, id, *origin, unnamed, error));
  CHECK(error == "signal name is required");
  CHECK(undo.count() == 0);
}

TEST_CASE("DBC edit message command applies undo redo and preserves signals") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SourceSet{0}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ flag : 8|8@1+ (1,0) [0|255] "" XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->sig("speed") != nullptr);
  REQUIRE(msg->sig("flag") != nullptr);

  loggy::Msg edited = *msg;
  edited.name = "RENAMED_MSG";
  edited.size = 3;
  edited.transmitter = "ECU";
  edited.comment = "edited";
  REQUIRE(loggy::commit_message_edit(undo, manager, id, edited, error));
  CHECK(error.empty());
  CHECK(undo.count() == 1);
  CHECK(msg->name == "RENAMED_MSG");
  CHECK(msg->size == 3);
  CHECK(msg->transmitter == "ECU");
  CHECK(msg->comment == "edited");
  REQUIRE(msg->sig("speed") != nullptr);
  REQUIRE(msg->sig("flag") != nullptr);

  undo.undo();
  CHECK(msg->name == "TEST_MSG");
  CHECK(msg->size == 2);
  CHECK(msg->transmitter == "XXX");
  CHECK(msg->comment.empty());
  REQUIRE(msg->sig("speed") != nullptr);
  REQUIRE(msg->sig("flag") != nullptr);

  undo.redo();
  CHECK(msg->name == "RENAMED_MSG");
  CHECK(msg->size == 3);
  CHECK(msg->transmitter == "ECU");
  CHECK(msg->comment == "edited");
  REQUIRE(msg->sig("speed") != nullptr);
  REQUIRE(msg->sig("flag") != nullptr);
}

TEST_CASE("DBC edit message command rejects invalid names sizes and shrink fit") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SourceSet{0}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ wide : 8|8@1+ (1,0) [0|255] "" XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);

  loggy::Msg empty_name = *msg;
  empty_name.name.clear();
  REQUIRE_FALSE(loggy::commit_message_edit(undo, manager, id, empty_name, error));
  CHECK(error == "message name is required");
  CHECK(undo.count() == 0);

  loggy::Msg invalid_size = *msg;
  invalid_size.size = 0;
  REQUIRE_FALSE(loggy::commit_message_edit(undo, manager, id, invalid_size, error));
  CHECK(error == "message size must be 1-64 bytes");
  CHECK(undo.count() == 0);

  loggy::Msg too_small = *msg;
  too_small.size = 1;
  REQUIRE_FALSE(loggy::commit_message_edit(undo, manager, id, too_small, error));
  CHECK(error == "message size would exclude signal: wide");
  CHECK(undo.count() == 0);
}

TEST_CASE("DBC add signal command creates signal and supports undo redo") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SOURCE_ALL, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Signal draft;
  draft.start_bit = 2;
  draft.size = 4;
  draft.is_little_endian = true;
  draft.factor = 1.0;
  draft.offset = 0.0;
  draft.min = 0.0;
  REQUIRE(loggy::commit_signal_add(undo, manager, id, draft, 2, error));
  CHECK(error.empty());
  CHECK(undo.count() == 1);
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  CHECK(msg->name == "NEW_MSG_123");
  CHECK(msg->size == 2);
  REQUIRE(msg->sig("NEW_SIGNAL_1") != nullptr);
  CHECK(msg->sig("NEW_SIGNAL_1")->start_bit == 2);
  CHECK(msg->sig("NEW_SIGNAL_1")->size == 4);
  CHECK(msg->sig("NEW_SIGNAL_1")->max == 15.0);

  undo.undo();
  CHECK(manager.msg(id) == nullptr);

  undo.redo();
  msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->sig("NEW_SIGNAL_1") != nullptr);
  CHECK(msg->sig("NEW_SIGNAL_1")->start_bit == 2);
}

TEST_CASE("DBC remove signal command applies undo and redo") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SourceSet{0}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ speed : 0|8@1+ (1,0) [0|255] "kph" XXX
 SG_ flag : 8|1@1+ (1,0) [0|1] "" XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  REQUIRE(loggy::commit_signal_remove(undo, manager, id, "flag", error));
  CHECK(error.empty());
  CHECK(undo.count() == 1);
  CHECK(msg->sig("flag") == nullptr);
  REQUIRE(msg->sig("speed") != nullptr);

  undo.undo();
  REQUIRE(msg->sig("flag") != nullptr);
  CHECK(msg->signals().size() == 2);

  undo.redo();
  CHECK(msg->sig("flag") == nullptr);
  CHECK(msg->signals().size() == 1);
}

TEST_CASE("DBC remove multiplexor command removes multiplexed children") {
  loggy::DBCManager manager;
  std::string error;
  REQUIRE(manager.open(loggy::SourceSet{0}, "inline", R"(
VERSION ""
NS_ :
BS_:
BU_: XXX
BO_ 291 TEST_MSG: 2 XXX
 SG_ mux M : 0|4@1+ (1,0) [0|15] "" XXX
 SG_ gated M4 : 8|8@1+ (1,0) [0|255] "" XXX
 SG_ normal : 16|1@1+ (1,0) [0|1] "" XXX
)", error));

  loggy::UndoStack undo;
  const loggy::MessageId id{.source = 0, .address = 291};
  loggy::Msg *msg = manager.msg(id);
  REQUIRE(msg != nullptr);
  REQUIRE(msg->sig("mux") != nullptr);
  REQUIRE(msg->sig("gated") != nullptr);
  REQUIRE(loggy::commit_signal_remove(undo, manager, id, "mux", error));
  CHECK(error.empty());
  CHECK(msg->sig("mux") == nullptr);
  CHECK(msg->sig("gated") == nullptr);
  REQUIRE(msg->sig("normal") != nullptr);

  undo.undo();
  REQUIRE(msg->sig("mux") != nullptr);
  REQUIRE(msg->sig("gated") != nullptr);
  REQUIRE(msg->sig("normal") != nullptr);
  CHECK(msg->sig("mux")->type == loggy::Signal::Type::Multiplexor);
  CHECK(msg->sig("gated")->type == loggy::Signal::Type::Multiplexed);
}
