#pragma once

#include <map>
#include <QList>
#include <QMetaType>
#include <QObject>
#include <QString>

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

  inline QString name() const { return name_; }
  void updateMsg(const MessageId &id, const QString &name, uint32_t size);
  void removeMsg(const MessageId &id);
  inline const std::map<uint32_t, cabana::Msg> &messages() const { return msgs; }
  inline const cabana::Msg *msg(const MessageId &id) const { return msg(id.address); }
  inline const cabana::Msg *msg(uint32_t address) const {
    auto it = msgs.find(address);
    return it != msgs.end() ? &it->second : nullptr;
  }

signals:
  void signalAdded(uint32_t address, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(uint32_t address);
  void msgRemoved(uint32_t address);
  void DBCFileChanged();

private:
  void parseExtraInfo(const QString &content);
  std::map<uint32_t, cabana::Msg> msgs;
  QString name_;
};

DBCManager *dbc();

inline QString msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}
