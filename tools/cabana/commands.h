#pragma once

#include <utility>

#include <QUndoCommand>
#include <QUndoStack>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class EditMsgCommand : public QUndoCommand {
public:
  EditMsgCommand(const MessageId &id, const QString &name, int size, const QString &node,
                 const QString &comment, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  QString old_name, new_name, old_comment, new_comment, old_node, new_node;
  int old_size = 0, new_size = 0;
};

class RemoveMsgCommand : public QUndoCommand {
public:
  RemoveMsgCommand(const MessageId &id, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  cabana::Msg message;
};

class AddSigCommand : public QUndoCommand {
public:
  AddSigCommand(const MessageId &id, const cabana::Signal &sig, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  bool msg_created = false;
  cabana::Signal signal = {};
};

class RemoveSigCommand : public QUndoCommand {
public:
  RemoveSigCommand(const MessageId &id, const cabana::Signal *sig, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  QList<cabana::Signal> sigs;
};

class EditSignalCommand : public QUndoCommand {
public:
  EditSignalCommand(const MessageId &id, const cabana::Signal *sig, const cabana::Signal &new_sig, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const MessageId id;
  QList<std::pair<cabana::Signal, cabana::Signal>> sigs; // QList<{old_sig, new_sig}>
};

namespace UndoStack {
  QUndoStack *instance();
  inline void push(QUndoCommand *cmd) { instance()->push(cmd); }
};
