#include "tools/loggy/backend/undo.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace loggy {
namespace {

std::string command_message_label(DBCManager *manager, MessageId id) {
  const Msg *msg = manager == nullptr ? nullptr : manager->msg(id);
  return (msg == nullptr || msg->name.empty()) ? std::string(UNTITLED) : msg->name;
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
  if (!canUndo()) return;
  commands_[static_cast<size_t>(--index_)]->undo();
}

void UndoStack::redo() {
  if (!canRedo()) return;
  commands_[static_cast<size_t>(index_++)]->redo();
}

void UndoStack::setIndex(int index) {
  index = std::clamp(index, 0, count());
  while (index_ < index) commands_[static_cast<size_t>(index_++)]->redo();
  while (index_ > index) commands_[static_cast<size_t>(--index_)]->undo();
}

void UndoStack::clear() {
  commands_.clear();
  index_ = 0;
  clean_index_ = 0;
}

void UndoStack::setClean() {
  clean_index_ = index_;
}

EditSignalCommand::EditSignalCommand(DBCManager *manager, MessageId id, const Signal *origin, Signal edited)
    : manager_(manager), id_(id) {
  if (manager_ == nullptr || origin == nullptr) return;
  edits_.push_back({*origin, std::move(edited)});
  if (origin->type == Signal::Type::Multiplexor && edits_.front().second.type == Signal::Type::Normal) {
    Msg *msg = manager_->msg(id_);
    assert(msg != nullptr);
    for (const Signal *signal : msg->getSignals()) {
      if (signal == nullptr || signal->type != Signal::Type::Multiplexed) continue;
      Signal normal = *signal;
      normal.type = Signal::Type::Normal;
      edits_.push_back({*signal, std::move(normal)});
    }
  }
  setText("edit signal " + origin->name + " in " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void EditSignalCommand::undo() {
  if (manager_ == nullptr) return;
  for (const auto &[old_signal, new_signal] : edits_) {
    manager_->updateSignal(id_, new_signal.name, old_signal);
  }
}

void EditSignalCommand::redo() {
  if (manager_ == nullptr) return;
  for (const auto &[old_signal, new_signal] : edits_) {
    manager_->updateSignal(id_, old_signal.name, new_signal);
  }
}

AddSignalCommand::AddSignalCommand(DBCManager *manager, MessageId id, Signal signal, uint32_t msg_size)
    : manager_(manager), id_(id), signal_(std::move(signal)), msg_size_(msg_size) {
  setText("add signal to " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void AddSignalCommand::undo() {
  if (manager_ == nullptr) return;
  manager_->removeSignal(id_, signal_.name);
  if (msg_created_) manager_->removeMsg(id_);
}

void AddSignalCommand::redo() {
  if (manager_ == nullptr) return;
  if (manager_->msg(id_) == nullptr) {
    msg_created_ = true;
    manager_->updateMsg(id_, manager_->newMsgName(id_), msg_size_, "", "");
  }
  if (signal_.name.empty()) {
    signal_.name = manager_->newSignalName(id_);
    setText("add signal " + signal_.name + " to " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
  }
  signal_.max = std::pow(2.0, static_cast<double>(signal_.size)) - 1.0;
  signal_.update();
  manager_->addSignal(id_, signal_);
}

RemoveSignalCommand::RemoveSignalCommand(DBCManager *manager, MessageId id, const Signal *origin)
    : manager_(manager), id_(id) {
  if (manager_ == nullptr || origin == nullptr) return;
  removed_.push_back(*origin);
  if (origin->type == Signal::Type::Multiplexor) {
    Msg *msg = manager_->msg(id_);
    assert(msg != nullptr);
    for (const Signal *signal : msg->getSignals()) {
      if (signal == nullptr || signal->type != Signal::Type::Multiplexed) continue;
      removed_.push_back(*signal);
    }
  }
  setText("remove signal " + origin->name + " from " + command_message_label(manager_, id_) + ":" + std::to_string(id_.address));
}

void RemoveSignalCommand::undo() {
  if (manager_ == nullptr) return;
  for (const Signal &signal : removed_) {
    manager_->addSignal(id_, signal);
  }
}

void RemoveSignalCommand::redo() {
  if (manager_ == nullptr) return;
  for (const Signal &signal : removed_) {
    manager_->removeSignal(id_, signal.name);
  }
}

bool commit_signal_add(UndoStack *undo_stack, DBCManager *manager, MessageId id,
                       Signal signal, uint32_t msg_size, std::string *error) {
  if (undo_stack == nullptr) {
    if (error != nullptr) *error = "undo stack is unavailable";
    return false;
  }
  if (manager == nullptr) {
    if (error != nullptr) *error = "DBC manager is unavailable";
    return false;
  }
  if (manager->findDBCFile(id.source) == nullptr) {
    if (error != nullptr) *error = "no DBC loaded for " + id.toString();
    return false;
  }
  if (Msg *msg = manager->msg(id); msg != nullptr) {
    msg_size = msg->size;
  }
  if (msg_size == 0 || msg_size > CAN_MAX_DATA_BYTES) {
    if (error != nullptr) *error = "message size must be 1-64 bytes";
    return false;
  }
  const int msg_bits = static_cast<int>(msg_size * 8);
  if (signal.size <= 0 || signal.size > msg_bits) {
    if (error != nullptr) *error = "signal size must be 1-" + std::to_string(msg_bits) + " bits";
    return false;
  }
  if (signal.start_bit < 0 || signal.start_bit >= msg_bits || signal.start_bit + signal.size > msg_bits) {
    if (error != nullptr) *error = "signal bit range must fit in the message";
    return false;
  }
  if (!signal.name.empty()) {
    if (Msg *msg = manager->msg(id); msg != nullptr && msg->sig(signal.name) != nullptr) {
      if (error != nullptr) *error = "duplicate signal name: " + signal.name;
      return false;
    }
  }
  undo_stack->push(std::make_unique<AddSignalCommand>(manager, id, std::move(signal), msg_size));
  if (error != nullptr) error->clear();
  return true;
}

bool commit_signal_edit(UndoStack *undo_stack, DBCManager *manager, MessageId id,
                        const Signal *origin, Signal edited, std::string *error) {
  if (undo_stack == nullptr) {
    if (error != nullptr) *error = "undo stack is unavailable";
    return false;
  }
  if (manager == nullptr || origin == nullptr) {
    if (error != nullptr) *error = "no DBC signal selected";
    return false;
  }
  Msg *msg = manager->msg(id);
  if (msg == nullptr) {
    if (error != nullptr) *error = "no DBC message for " + id.toString();
    return false;
  }
  if (edited.name.empty()) {
    if (error != nullptr) *error = "signal name is required";
    return false;
  }
  if (edited.size <= 0 || edited.size > CAN_MAX_DATA_BYTES * 8) {
    if (error != nullptr) *error = "signal size must be 1-512 bits";
    return false;
  }
  if (edited.start_bit < 0 || edited.start_bit >= CAN_MAX_DATA_BYTES * 8) {
    if (error != nullptr) *error = "signal start bit must be 0-511";
    return false;
  }
  if (edited.name != origin->name && msg->sig(edited.name) != nullptr) {
    if (error != nullptr) *error = "duplicate signal name: " + edited.name;
    return false;
  }
  if (edited.is_little_endian != origin->is_little_endian) {
    edited.start_bit = flipBitPos(edited.start_bit);
  }
  edited.update();
  undo_stack->push(std::make_unique<EditSignalCommand>(manager, id, origin, std::move(edited)));
  if (error != nullptr) error->clear();
  return true;
}

bool commit_signal_remove(UndoStack *undo_stack, DBCManager *manager, MessageId id,
                          const std::string &signal_name, std::string *error) {
  if (undo_stack == nullptr) {
    if (error != nullptr) *error = "undo stack is unavailable";
    return false;
  }
  if (manager == nullptr) {
    if (error != nullptr) *error = "DBC manager is unavailable";
    return false;
  }
  Msg *msg = manager->msg(id);
  if (msg == nullptr) {
    if (error != nullptr) *error = "no DBC message for " + id.toString();
    return false;
  }
  const Signal *origin = msg->sig(signal_name);
  if (origin == nullptr) {
    if (error != nullptr) *error = "signal no longer exists: " + signal_name;
    return false;
  }
  undo_stack->push(std::make_unique<RemoveSignalCommand>(manager, id, origin));
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace loggy
