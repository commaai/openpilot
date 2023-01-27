#include <QApplication>

#include "tools/cabana/commands.h"

// EditMsgCommand

EditMsgCommand::EditMsgCommand(const QString &id, const QString &title, int size, QUndoCommand *parent)
    : id(id), new_title(title), new_size(size), QUndoCommand(parent) {
  if (auto msg = dbc()->msg(id)) {
    old_title = msg->name;
    old_size = msg->size;
  }
  setText(QObject::tr("Edit message %1:%2").arg(DBCManager::parseId(id).second).arg(title));
}

void EditMsgCommand::undo() {
  if (old_title.isEmpty())
    dbc()->removeMsg(id);
  else
    dbc()->updateMsg(id, old_title, old_size);
}

void EditMsgCommand::redo() {
  dbc()->updateMsg(id, new_title, new_size);
}

// RemoveMsgCommand

RemoveMsgCommand::RemoveMsgCommand(const QString &id, QUndoCommand *parent) : id(id), QUndoCommand(parent) {
  if (auto msg = dbc()->msg(id)) {
    message = *msg;
    setText(QObject::tr("Remove message %1:%2").arg(DBCManager::parseId(id).second).arg(message.name));
  }
}

void RemoveMsgCommand::undo() {
  if (!message.name.isEmpty()) {
    dbc()->updateMsg(id, message.name, message.size);
    for (auto &[name, s] : message.sigs)
      dbc()->addSignal(id, s);
  }
}

void RemoveMsgCommand::redo() {
  if (!message.name.isEmpty())
    dbc()->removeMsg(id);
}

// AddSigCommand

AddSigCommand::AddSigCommand(const QString &id, const Signal &sig, QUndoCommand *parent)
    : id(id), signal(sig), QUndoCommand(parent) {
  setText(QObject::tr("Add signal %1 to %2").arg(sig.name.c_str()).arg(DBCManager::parseId(id).second));
}

void AddSigCommand::undo() { dbc()->removeSignal(id, signal.name.c_str()); }
void AddSigCommand::redo() { dbc()->addSignal(id, signal); }

// RemoveSigCommand

RemoveSigCommand::RemoveSigCommand(const QString &id, const Signal *sig, QUndoCommand *parent)
    : id(id), signal(*sig), QUndoCommand(parent) {
  setText(QObject::tr("Remove signal %1 from %2").arg(signal.name.c_str()).arg(DBCManager::parseId(id).second));
}

void RemoveSigCommand::undo() { dbc()->addSignal(id, signal); }
void RemoveSigCommand::redo() { dbc()->removeSignal(id, signal.name.c_str()); }

// EditSignalCommand

EditSignalCommand::EditSignalCommand(const QString &id, const Signal *sig, const Signal &new_sig, QUndoCommand *parent)
    : id(id), old_signal(*sig), new_signal(new_sig), QUndoCommand(parent) {
  setText(QObject::tr("Edit signal %1").arg(old_signal.name.c_str()));
}

void EditSignalCommand::undo() { dbc()->updateSignal(id, new_signal.name.c_str(), old_signal); }
void EditSignalCommand::redo() { dbc()->updateSignal(id, old_signal.name.c_str(), new_signal); }

namespace UndoStack {

QUndoStack *instance() {
  static QUndoStack *undo_stack = new QUndoStack(qApp);
  return undo_stack;
}

}  // namespace UndoStack
