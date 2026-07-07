#include "tools/cabana/commands.h"

#include <algorithm>
#include <cassert>
#include <cmath>

// EditMsgCommand

EditMsgCommand::EditMsgCommand(const MessageId &id, const std::string &name, int size,
                               const std::string &node, const std::string &comment)
    : id(id), new_name(name), new_size(size), new_node(node), new_comment(comment) {
  if (auto msg = dbc()->msg(id)) {
    old_name = msg->name;
    old_size = msg->size;
    old_node = msg->transmitter;
    old_comment = msg->comment;
    setText("edit message " + name + ":" + std::to_string(id.address));
  } else {
    setText("new message " + name + ":" + std::to_string(id.address));
  }
}

void EditMsgCommand::undo() {
  if (old_name.empty())
    dbc()->removeMsg(id);
  else
    dbc()->updateMsg(id, old_name, old_size, old_node, old_comment);
}

void EditMsgCommand::redo() {
  dbc()->updateMsg(id, new_name, new_size, new_node, new_comment);
}

// RemoveMsgCommand

RemoveMsgCommand::RemoveMsgCommand(const MessageId &id) : id(id) {
  if (auto msg = dbc()->msg(id)) {
    message = *msg;
    setText("remove message " + message.name + ":" + std::to_string(id.address));
  }
}

void RemoveMsgCommand::undo() {
  if (!message.name.empty()) {
    dbc()->updateMsg(id, message.name, message.size, message.transmitter, message.comment);
    for (auto s : message.getSignals())
      dbc()->addSignal(id, *s);
  }
}

void RemoveMsgCommand::redo() {
  if (!message.name.empty())
    dbc()->removeMsg(id);
}

// AddSigCommand

AddSigCommand::AddSigCommand(const MessageId &id, const cabana::Signal &sig, int msg_size)
    : id(id), msg_size(msg_size), signal(sig) {
  setText("add signal " + sig.name + " to " + msgName(id) + ":" + std::to_string(id.address));
}

void AddSigCommand::undo() {
  dbc()->removeSignal(id, signal.name);
  if (msg_created) dbc()->removeMsg(id);
}

void AddSigCommand::redo() {
  if (auto msg = dbc()->msg(id); !msg) {
    msg_created = true;
    dbc()->updateMsg(id, dbc()->newMsgName(id), msg_size, "", "");
  }
  signal.name = dbc()->newSignalName(id);
  signal.max = std::pow(2, signal.size) - 1;
  dbc()->addSignal(id, signal);
}

// RemoveSigCommand

RemoveSigCommand::RemoveSigCommand(const MessageId &id, const cabana::Signal *sig) : id(id) {
  sigs.push_back(*sig);
  if (sig->type == cabana::Signal::Type::Multiplexor) {
    for (const auto &s : dbc()->msg(id)->sigs) {
      if (s->type == cabana::Signal::Type::Multiplexed) {
        sigs.push_back(*s);
      }
    }
  }
  setText("remove signal " + sig->name + " from " + msgName(id) + ":" + std::to_string(id.address));
}

void RemoveSigCommand::undo() { for (const auto &s : sigs) dbc()->addSignal(id, s); }
void RemoveSigCommand::redo() { for (const auto &s : sigs) dbc()->removeSignal(id, s.name); }

// EditSignalCommand

EditSignalCommand::EditSignalCommand(const MessageId &id, const cabana::Signal *sig, const cabana::Signal &new_sig)
    : id(id) {
  sigs.push_back({*sig, new_sig});
  if (sig->type == cabana::Signal::Type::Multiplexor && new_sig.type == cabana::Signal::Type::Normal) {
    // convert all multiplexed signals to normal signals
    auto msg = dbc()->msg(id);
    assert(msg);
    for (const auto &s : msg->sigs) {
      if (s->type == cabana::Signal::Type::Multiplexed) {
        auto new_s = *s;
        new_s.type = cabana::Signal::Type::Normal;
        sigs.push_back({*s, new_s});
      }
    }
  }
  setText("edit signal " + sig->name + " in " + msgName(id) + ":" + std::to_string(id.address));
}

void EditSignalCommand::undo() { for (const auto &s : sigs) dbc()->updateSignal(id, s.second.name, s.first); }
void EditSignalCommand::redo() { for (const auto &s : sigs) dbc()->updateSignal(id, s.first.name, s.second); }

// UndoStack

UndoStack *UndoStack::instance() {
  static UndoStack stack;
  return &stack;
}

void UndoStack::pushCommand(UndoCommand *cmd) {
  bool was_clean = isClean();
  cmd->redo();
  commands_.resize(index_);
  if (clean_index_ > index_) clean_index_ = -1;
  commands_.emplace_back(cmd);
  ++index_;
  indexChanged(index_);
  if (isClean() != was_clean) cleanChanged(isClean());
}

void UndoStack::undo() {
  if (!canUndo()) return;
  bool was_clean = isClean();
  commands_[--index_]->undo();
  indexChanged(index_);
  if (isClean() != was_clean) cleanChanged(isClean());
}

void UndoStack::redo() {
  if (!canRedo()) return;
  bool was_clean = isClean();
  commands_[index_++]->redo();
  indexChanged(index_);
  if (isClean() != was_clean) cleanChanged(isClean());
}

void UndoStack::setIndex(int i) {
  i = std::clamp(i, 0, count());
  if (i == index_) return;
  bool was_clean = isClean();
  while (index_ < i) commands_[index_++]->redo();
  while (index_ > i) commands_[--index_]->undo();
  indexChanged(index_);
  if (isClean() != was_clean) cleanChanged(isClean());
}

void UndoStack::clear() {
  bool was_clean = isClean();
  commands_.clear();
  index_ = 0;
  clean_index_ = 0;
  indexChanged(index_);
  if (!was_clean) cleanChanged(true);
}

void UndoStack::setClean() {
  bool was_clean = isClean();
  clean_index_ = index_;
  if (!was_clean) cleanChanged(true);
}
