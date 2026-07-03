#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/utils/event.h"

class UndoCommand {
public:
  virtual ~UndoCommand() = default;
  virtual void undo() = 0;
  virtual void redo() = 0;
  const std::string &text() const { return text_; }

protected:
  void setText(const std::string &t) { text_ = t; }

private:
  std::string text_;
};

class UndoStack {
public:
  static UndoStack *instance();
  static void push(UndoCommand *cmd) { instance()->pushCommand(cmd); }

  void pushCommand(UndoCommand *cmd);
  void undo();
  void redo();
  bool canUndo() const { return index_ > 0; }
  bool canRedo() const { return index_ < (int)commands_.size(); }
  int index() const { return index_; }
  int count() const { return (int)commands_.size(); }
  const std::string &text(int i) const { return commands_[i]->text(); }
  std::string undoText() const { return canUndo() ? commands_[index_ - 1]->text() : std::string(); }
  std::string redoText() const { return canRedo() ? commands_[index_]->text() : std::string(); }
  void setIndex(int i);
  void clear();
  bool isClean() const { return index_ == clean_index_; }
  void setClean();

  cabana::Event<int> indexChanged;
  cabana::Event<bool> cleanChanged;

private:
  std::vector<std::unique_ptr<UndoCommand>> commands_;
  int index_ = 0;
  int clean_index_ = 0;
};

class EditMsgCommand : public UndoCommand {
public:
  EditMsgCommand(const MessageId &id, const std::string &name, int size, const std::string &node, const std::string &comment);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  std::string old_name, new_name, old_comment, new_comment, old_node, new_node;
  int old_size = 0, new_size = 0;
};

class RemoveMsgCommand : public UndoCommand {
public:
  explicit RemoveMsgCommand(const MessageId &id);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  cabana::Msg message;
};

class AddSigCommand : public UndoCommand {
public:
  AddSigCommand(const MessageId &id, const cabana::Signal &sig, int msg_size);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  int msg_size;
  bool msg_created = false;
  cabana::Signal signal = {};
};

class RemoveSigCommand : public UndoCommand {
public:
  RemoveSigCommand(const MessageId &id, const cabana::Signal *sig);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  std::vector<cabana::Signal> sigs;
};

class EditSignalCommand : public UndoCommand {
public:
  EditSignalCommand(const MessageId &id, const cabana::Signal *sig, const cabana::Signal &new_sig);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  std::vector<std::pair<cabana::Signal, cabana::Signal>> sigs; // {old_sig, new_sig}
};
