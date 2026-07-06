#pragma once

#include "tools/loggy/backend/dbc/dbcmanager.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace loggy {

class UndoCommand {
public:
  virtual ~UndoCommand() = default;
  virtual void undo() = 0;
  virtual void redo() = 0;
  const std::string &text() const { return text_; }

protected:
  void set_text(std::string text) { text_ = std::move(text); }

private:
  std::string text_;
};

struct UndoStackEntry {
  int index = 0;
  std::string text;
  bool applied = false;
  bool next = false;
  bool clean = false;
};

class UndoStack {
public:
  void push(std::unique_ptr<UndoCommand> command);
  void undo();
  void redo();
  void set_index(int index);
  void clear();
  void set_clean();

  bool can_undo() const { return index_ > 0; }
  bool can_redo() const { return index_ < static_cast<int>(commands_.size()); }
  int index() const { return index_; }
  int count() const { return static_cast<int>(commands_.size()); }
  bool is_clean() const { return index_ == clean_index_; }
  std::string undo_text() const { return can_undo() ? commands_[static_cast<size_t>(index_ - 1)]->text() : std::string(); }
  std::string redo_text() const { return can_redo() ? commands_[static_cast<size_t>(index_)]->text() : std::string(); }
  std::vector<UndoStackEntry> entries() const;

private:
  std::vector<std::unique_ptr<UndoCommand>> commands_;
  int index_ = 0;
  int clean_index_ = 0;
};

class EditSignalCommand : public UndoCommand {
public:
  EditSignalCommand(DBCManager &manager, MessageId id, const Signal &origin, Signal edited);
  void undo() override;
  void redo() override;

private:
  DBCManager &manager_;
  MessageId id_;
  std::vector<std::pair<Signal, Signal>> edits_;
};

class AddSignalCommand : public UndoCommand {
public:
  AddSignalCommand(DBCManager &manager, MessageId id, Signal signal, uint32_t msg_size);
  void undo() override;
  void redo() override;

private:
  DBCManager &manager_;
  MessageId id_;
  Signal signal_;
  uint32_t msg_size_ = 0;
  bool msg_created_ = false;
};

class RemoveSignalCommand : public UndoCommand {
public:
  RemoveSignalCommand(DBCManager &manager, MessageId id, const Signal &origin);
  void undo() override;
  void redo() override;

private:
  DBCManager &manager_;
  MessageId id_;
  std::vector<Signal> removed_;
};

class EditMessageCommand : public UndoCommand {
public:
  EditMessageCommand(DBCManager &manager, MessageId id, Msg origin, Msg edited);
  void undo() override;
  void redo() override;

private:
  DBCManager &manager_;
  MessageId id_;
  Msg origin_;
  Msg edited_;
};

bool commit_signal_add(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                       Signal signal, uint32_t msg_size, std::string &error);
bool commit_signal_edit(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                        const Signal &origin, Signal edited, std::string &error);
bool commit_signal_remove(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                          const std::string &signal_name, std::string &error);
bool commit_message_edit(UndoStack &undo_stack, DBCManager &manager, MessageId id,
                         Msg edited, std::string &error);

}  // namespace loggy
