#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tools/cabana/core/observable.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class UndoCommand {
public:
  virtual ~UndoCommand() = default;
  virtual void undo() = 0;
  virtual void redo() = 0;
  std::string text;
};

class UndoStack {
public:
  void push(UndoCommand *cmd);  // takes ownership and calls redo()
  void undo();
  void redo();
  void clear();
  void setClean();
  bool isClean() const { return clean_index_ == index_; }
  bool canUndo() const { return index_ > 0; }
  bool canRedo() const { return index_ < (int)commands_.size(); }
  std::string undoText() const { return canUndo() ? commands_[index_ - 1]->text : ""; }
  std::string redoText() const { return canRedo() ? commands_[index_]->text : ""; }
  static UndoStack *instance();

  Observable<> indexChanged;
  Observable<bool> cleanChanged;

private:
  void setIndex(int index);
  std::vector<std::unique_ptr<UndoCommand>> commands_;
  int index_ = 0;
  int clean_index_ = 0;
};

class EditMsgCommand : public UndoCommand {
public:
  EditMsgCommand(const MessageId &id, const std::string &name, int size, const std::string &node,
                 const std::string &comment);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  std::string old_name, new_name, old_comment, new_comment, old_node, new_node;
  int old_size = 0, new_size = 0;
};

class RemoveMsgCommand : public UndoCommand {
public:
  RemoveMsgCommand(const MessageId &id);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  cabana::Msg message;
};

class AddSigCommand : public UndoCommand {
public:
  AddSigCommand(const MessageId &id, const cabana::Signal &sig);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
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
