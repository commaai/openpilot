#pragma once

#include <map>
#include <QList>
#include <QMetaType>
#include <QObject>
#include <QString>
#include <QSet>
#include <QDebug>

#include "tools/cabana/dbc.h"

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent) {}
  ~DBCManager() {}
  bool open(const QString &dbc_file_name, QString *error = nullptr);
  bool open(const QString &name, const QString &content, QString *error = nullptr);
  QString generateDBC();
  void addSignal(const MessageId &id, const cabana::Signal &sig);
  void updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig);
  void removeSignal(const MessageId &id, const QString &sig_name);
  inline int msgCount() const { return msgs.size(); }

  inline QString name() const { return name_; }
  void updateMsg(const MessageId &id, const QString &name, uint32_t size);
  void removeMsg(const MessageId &id);
  inline std::map<MessageId, cabana::Msg> getMessages(uint8_t source) {
    std::map<MessageId, cabana::Msg> ret;
    for (auto &[address, msg] : msgs) {
      MessageId id = {.source = source, .address = address};
      ret[id] = msg;
    }
    return ret;
  }
  inline const cabana::Msg *msg(const MessageId &id) const { return msg(id.address); }
  inline const cabana::Msg* msg(uint8_t source, const QString &name) {
    for (auto &[_, msg] : msgs) {
      if (msg.name == name) {
        return &msg;
      }
    }

    return nullptr;
  }
  QStringList signalNames();

public slots:
  void updateSources(const QSet<uint8_t> &s);

signals:
  void signalAdded(MessageId id, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id);
  void DBCFileChanged();

private:
  void parseExtraInfo(const QString &content);
  std::map<uint32_t, cabana::Msg> msgs;
  QString name_;
  QSet<uint8_t> sources;

  inline const cabana::Msg *msg(uint32_t address) const {
    auto it = msgs.find(address);
    return it != msgs.end() ? &it->second : nullptr;
  }
};

DBCManager *dbc();

inline QString msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}
