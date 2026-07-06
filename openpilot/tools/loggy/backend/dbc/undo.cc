#include "tools/loggy/backend/dbc/undo.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace loggy {
namespace {

std::string command_message_label(DBCManager &manager, MessageId id) {
  const Msg *msg = manager.msg(id);
  return (msg == nullptr || msg->name.empty()) ? std::string(UNTITLED) : msg->name;
}

bool signal_fits_message_size(const Signal &signal, uint32_t msg_size) {
  Signal normalized = signal;
  normalized.update();
  const int msg_bits = static_cast<int>(msg_size * 8);
  return normalized.size > 0 &&
         normalized.lsb >= 0 &&
         normalized.msb >= 0 &&
         normalized.lsb < msg_bits &&
         normalized.msb < msg_bits;
}

}  // namespace

void UndoStack::push(std::unique_ptr<UndoCommand> command) {
  if (!command) return;
  command->redo();
  commands_.resize(static_cast<size_t>(index_));
  if (clean_index_ > index_) clean_index_ = -1;
  commands_.push_back(std::move(command));
  ++index_;
}

void UndoStack::undo() {
  if (!can_undo()) return;
  commands_[static_cast<size_t>(--index_)]->undo();
}

void UndoStack::redo() {
  if (!can_redo()) return;
  commands_[static_cast<size_t>(index_++)]->redo();
}

void UndoStack::set_index(int index) {
  index = std::clamp(index, 0, count());
  while (index_ < index) commands_[static_cast<size_t>(index_++)]->redo();
  while (index_ > index) commands_[static_cast<size_t>(--index_)]->undo();
}

void UndoStack::clear() {
  commands_.clear();
  index_ = 0;
  clean_index_ = 0;
}

void UndoStack::set_clean() {
  clean_index_ = index_;
}

std::vector<UndoStackEntry> UndoStack::entries() const {
  std::vector<UndoStackEntry> rows;
  rows.reserve(commands_.size());
  for (size_t i = 0; i < commands_.size(); ++i) {
    rows.push_back(UndoStackEntry{
      .index = static_cast<int>(i),
      .text = commands_[i] ? commands_[i]->text() : std::string(),
      .applied = static_cast<int>(i) < index_,
      .next = static_cast<int>(i) == index_,
      .clean = clean_index_ == static_cast<int>(i) + 1,
    });
  }
  return rows;
}

EditSignalCommand::EditSignalCommand(DBCManager &manager, MessageId id, const Signal &origin, Signal edited)
    : manager_(manager), id_(id) {
  edits_.push_back({origin, std::move(edited)});
  if (origin.type == Signal::Type::Multiplexor && edits_.front().second.type == Signal::Type::Normal) {
    Msg *msg = manager_.msg(id_);
    assert(msg != nullptr);
    for (const Signal *signal : msg->signals()) {
      if (signal == nullptr || signal->type != Signal::Type::Multiplexed) continue;
      Signal normal = *signal;
      normal.type = Signal::Type::Normal;
      edits_.push_back({*signal, std::move(normal)});
    }
  }
  setText("edit signal " + origin.name + " in " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void EditSignalCommand::undo() {
  for (const auto &[old_signal, new_signal] : edits_) {
    manager_.updateSignal(id_, new_signal.name, old_signal);
  }
}

void EditSignalCommand::redo() {
  for (const auto &[old_signal, new_signal] : edits_) {
    manager_.updateSignal(id_, old_signal.name, new_signal);
  }
}

AddSignalCommand::AddSignalCommand(DBCManager &manager, MessageId id, Signal signal, uint32_t msg_size)
    : manager_(manager), id_(id), signal_(std::move(signal)), msg_size_(msg_size) {
  setText("add signal to " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void AddSignalCommand::undo() {
  manager_.removeSignal(id_, signal_.name);
  if (msg_created_) manager_.remove_msg(id_);
}

void AddSignalCommand::redo() {
  if (manager_.msg(id_) == nullptr) {
    msg_created_ = true;
    manager_.update_msg(id_, manager_.new_msg_name(id_), msg_size_, "", "");
  }
  if (signal_.name.empty()) {
    signal_.name = manager_.new_signal_name(id_);
    setText("add signal " + signal_.name + " to " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
  }
  signal_.max = std::pow(2.0, static_cast<double>(signal_.size)) - 1.0;
  signal_.update();
  manager_.add_signal(id_, signal_);
}

RemoveSignalCommand::RemoveSignalCommand(DBCManager &manager, MessageId id, const Signal &origin)
    : manager_(manager), id_(id) {
  removed_.push_back(origin);
  if (origin.type == Signal::Type::Multiplexor) {
    Msg *msg = manager_.msg(id_);
    assert(msg != nullptr);
    for (const Signal *signal : msg->signals()) {
      if (signal == nullptr || signal->type != Signal::Type::Multiplexed) continue;
      removed_.push_back(*signal);
    }
  }
  setText("remove signal " + origin.name + " from " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void RemoveSignalCommand::undo() {
  for (const Signal &signal : removed_) {
    manager_.add_signal(id_, signal);
  }
}

void RemoveSignalCommand::redo() {
  for (const Signal &signal : removed_) {
    manager_.removeSignal(id_, signal.name);
  }
}

EditMessageCommand::EditMessageCommand(DBCManager &manager, MessageId id, Msg origin, Msg edited)
    : manager_(manager), id_(id), origin_(std::move(origin)), edited_(std::move(edited)) {
  setText("edit message " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void EditMessageCommand::undo() {
  manager_.update_msg(id_, origin_.name, origin_.size, origin_.transmitter, origin_.comment);
}

void EditMessageCommand::redo() {
  manager_.update_msg(id_, edited_.name, edited_.size, edited_.transmitter, edited_.comment);
}

bool commit_signal_add(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                       Signal signal, uint32_t msg_size, std::string &error) {
  if (manager.find_dbc_file(id.source) == nullptr) {
    error = "no DBC loaded for " + id.to_string();
    return false;
  }
  if (Msg *msg = manager.msg(id); msg != nullptr) {
    msg_size = msg->size;
  }
  if (msg_size == 0 || msg_size > CAN_MAX_DATA_BYTES) {
    error = "message size must be 1-64 bytes";
    return false;
  }
  const int msg_bits = static_cast<int>(msg_size * 8);
  if (signal.size <= 0 || signal.size > msg_bits) {
    error = "signal size must be 1-" + std::to_string(msg_bits) + " bits";
    return false;
  }
  if (signal.start_bit < 0 || signal.start_bit >= msg_bits || signal.start_bit + signal.size > msg_bits) {
    error = "signal bit range must fit in the message";
    return false;
  }
  if (!signal.name.empty()) {
    if (Msg *msg = manager.msg(id); msg != nullptr && msg->sig(signal.name) != nullptr) {
      error = "duplicate signal name: " + signal.name;
      return false;
    }
  }
  undo_stack.push(std::make_unique<AddSignalCommand>(manager, id, std::move(signal), msg_size));
  error.clear();
  return true;
}

bool commit_signal_edit(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                        const Signal &origin, Signal edited, std::string &error) {
  Msg *msg = manager.msg(id);
  if (msg == nullptr) {
    error = "no DBC message for " + id.to_string();
    return false;
  }
  if (edited.name.empty()) {
    error = "signal name is required";
    return false;
  }
  if (edited.size <= 0 || edited.size > CAN_MAX_DATA_BYTES * 8) {
    error = "signal size must be 1-512 bits";
    return false;
  }
  if (edited.start_bit < 0 || edited.start_bit >= CAN_MAX_DATA_BYTES * 8) {
    error = "signal start_ bit must be 0-511";
    return false;
  }
  if (edited.name != origin.name && msg->sig(edited.name) != nullptr) {
    error = "duplicate signal name: " + edited.name;
    return false;
  }
  if (edited.is_little_endian != origin.is_little_endian) {
    edited.start_bit = flip_bit_pos(edited.start_bit);
  }
  edited.update();
  undo_stack.push(std::make_unique<EditSignalCommand>(manager, id, origin, std::move(edited)));
  error.clear();
  return true;
}

bool commit_signal_remove(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                          const std::string &signal_name, std::string &error) {
  Msg *msg = manager.msg(id);
  if (msg == nullptr) {
    error = "no DBC message for " + id.to_string();
    return false;
  }
  const Signal *origin = msg->sig(signal_name);
  if (origin == nullptr) {
    error = "signal no longer exists: " + signal_name;
    return false;
  }
  undo_stack.push(std::make_unique<RemoveSignalCommand>(manager, id, *origin));
  error.clear();
  return true;
}

bool commit_message_edit(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                         Msg edited, std::string &error) {
  Msg *origin = manager.msg(id);
  if (origin == nullptr) {
    error = "no DBC message for " + id.to_string();
    return false;
  }
  if (edited.name.empty()) {
    error = "message name is required";
    return false;
  }
  if (edited.size == 0 || edited.size > CAN_MAX_DATA_BYTES) {
    error = "message size must be 1-64 bytes";
    return false;
  }
  for (const Signal *signal : origin->signals()) {
    if (signal == nullptr) continue;
    if (!signal_fits_message_size(*signal, edited.size)) {
      error = "message size would exclude signal: " + signal->name;
      return false;
    }
  }
  edited.address = id.address;
  edited.transmitter = edited.transmitter.empty() ? DEFAULT_NODE_NAME : edited.transmitter;
  edited.update();
  Msg old_message = *origin;
  if (old_message.name == edited.name &&
      old_message.size == edited.size &&
      old_message.transmitter == edited.transmitter &&
      old_message.comment == edited.comment) {
    error.clear();
    return true;
  }
  undo_stack.push(std::make_unique<EditMessageCommand>(manager, id, std::move(old_message), std::move(edited)));
  error.clear();
  return true;
}

}  // namespace loggy
