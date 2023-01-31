#pragma once

#include <QUndoCommand>
#include <QUndoStack>

#include "tools/cabana/dbcmanager.h"

class EditMsgCommand : public QUndoCommand {
public:
  EditMsgCommand(const QString &id, const QString &title, int size, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const QString id;
  QString old_title, new_title;
  int old_size = 0, new_size = 0;
};

class RemoveMsgCommand : public QUndoCommand {
public:
  RemoveMsgCommand(const QString &id, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const QString id;
  DBCMsg message;
};

class AddSigCommand : public QUndoCommand {
public:
  AddSigCommand(const QString &id, const Signal &sig, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const QString id;
  Signal signal = {};
};

class RemoveSigCommand : public QUndoCommand {
public:
  RemoveSigCommand(const QString &id, const Signal *sig, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const QString id;
  Signal signal = {};
};

class EditSignalCommand : public QUndoCommand {
public:
  EditSignalCommand(const QString &id, const Signal *sig, const Signal &new_sig, QUndoCommand *parent = nullptr);
  void undo() override;
  void redo() override;

private:
  const QString id;
  Signal old_signal = {};
  Signal new_signal = {};
};

namespace UndoStack {
  QUndoStack *instance();
  inline void push(QUndoCommand *cmd) { instance()->push(cmd); }
};
