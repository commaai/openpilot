#include <QApplication>

#include "tools/cabana/commands.h"

// EditMsgCommand

EditMsgCommand::EditMsgCommand(const MessageId &id, const QString &name, int size,
                               const QString &node, const QString &comment, QUndoCommand *parent)
    : id(id), new_name(name), new_size(size), new_node(node), new_comment(comment), QUndoCommand(parent) {
  if (auto msg = dbc()->msg(id)) {
    old_name = msg->name;
    old_size = msg->size;
    old_node = msg->transmitter;
    old_comment = msg->comment;
    setText(QObject::tr("edit message %1:%2").arg(name).arg(id.address));
  } else {
    setText(QObject::tr("new message %1:%2").arg(name).arg(id.address));
  }
}

void EditMsgCommand::undo() {
  if (old_name.isEmpty())
    dbc()->removeMsg(id);
  else
    dbc()->updateMsg(id, old_name, old_size, old_node, old_comment);
}

void EditMsgCommand::redo() {
  dbc()->updateMsg(id, new_name, new_size, new_node, new_comment);
}

// RemoveMsgCommand

RemoveMsgCommand::RemoveMsgCommand(const MessageId &id, QUndoCommand *parent) : id(id), QUndoCommand(parent) {
  if (auto msg = dbc()->msg(id)) {
    message = *msg;
    setText(QObject::tr("remove message %1:%2").arg(message.name).arg(id.address));
  }
}

void RemoveMsgCommand::undo() {
  if (!message.name.isEmpty()) {
    dbc()->updateMsg(id, message.name, message.size, message.transmitter, message.comment);
    for (auto s : message.getSignals())
      dbc()->addSignal(id, *s);
  }
}

void RemoveMsgCommand::redo() {
  if (!message.name.isEmpty())
    dbc()->removeMsg(id);
}

// AddSigCommand

AddSigCommand::AddSigCommand(const MessageId &id, const cabana::Signal &sig, QUndoCommand *parent)
    : id(id), signal(sig), QUndoCommand(parent) {
  setText(QObject::tr("add signal %1 to %2:%3").arg(sig.name).arg(msgName(id)).arg(id.address));
}

void AddSigCommand::undo() {
  dbc()->removeSignal(id, signal.name);
  if (msg_created) dbc()->removeMsg(id);
}

void AddSigCommand::redo() {
  if (auto msg = dbc()->msg(id); !msg) {
    msg_created = true;
    dbc()->updateMsg(id, dbc()->newMsgName(id), can->lastMessage(id).dat.size(), "", "");
  }
  signal.name = dbc()->newSignalName(id);
  signal.max = std::pow(2, signal.size) - 1;
  dbc()->addSignal(id, signal);
}

// RemoveSigCommand

RemoveSigCommand::RemoveSigCommand(const MessageId &id, const cabana::Signal *sig, QUndoCommand *parent)
    : id(id), QUndoCommand(parent) {
  sigs.push_back(*sig);
  if (sig->type == cabana::Signal::Type::Multiplexor) {
    for (const auto &s : dbc()->msg(id)->sigs) {
      if (s->type == cabana::Signal::Type::Multiplexed) {
        sigs.push_back(*s);
      }
    }
  }
  setText(QObject::tr("remove signal %1 from %2:%3").arg(sig->name).arg(msgName(id)).arg(id.address));
}

void RemoveSigCommand::undo() { for (const auto &s : sigs) dbc()->addSignal(id, s); }
void RemoveSigCommand::redo() { for (const auto &s : sigs) dbc()->removeSignal(id, s.name); }

// EditSignalCommand

EditSignalCommand::EditSignalCommand(const MessageId &id, const cabana::Signal *sig, const cabana::Signal &new_sig, QUndoCommand *parent)
    : id(id), QUndoCommand(parent) {
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
  setText(QObject::tr("edit signal %1 in %2:%3").arg(sig->name).arg(msgName(id)).arg(id.address));
}

void EditSignalCommand::undo() { for (const auto &s : sigs) dbc()->updateSignal(id, s.second.name, s.first); }
void EditSignalCommand::redo() { for (const auto &s : sigs) dbc()->updateSignal(id, s.first.name, s.second); }

namespace UndoStack {

QUndoStack *instance() {
  static QUndoStack *undo_stack = new QUndoStack(qApp);
  return undo_stack;
}

}  // namespace UndoStack
