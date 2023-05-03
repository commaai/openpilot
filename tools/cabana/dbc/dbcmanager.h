#pragma once

#include <QList>
#include <QSet>
#include <QString>
#include <map>
#include <memory>
#include <set>

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcfile.h"

typedef QSet<int> SourceSet;
const SourceSet SOURCE_ALL = {-1};
inline bool operator<(const std::shared_ptr<DBCFile> &l, const std::shared_ptr<DBCFile> &r) { return l.get() < r.get(); }

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent) {}
  ~DBCManager() {}
  bool open(const SourceSet &source, const QString &dbc_file_name, QString *error = nullptr);
  bool open(const SourceSet &source, const QString &name, const QString &content, QString *error = nullptr);
  void close(const SourceSet &source);
  void closeAll();
  inline void close(DBCFile *file) { removeSourcesFromFile(file, {}); }
  void removeSourcesFromFile(DBCFile *file, const SourceSet &s);

  void addSignal(const MessageId &id, const cabana::Signal &sig);
  void updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig);
  void removeSignal(const MessageId &id, const QString &sig_name);

  void updateMsg(const MessageId &id, const QString &name, uint32_t size);
  void removeMsg(const MessageId &id);

  QString newMsgName(const MessageId &id);
  QString newSignalName(const MessageId &id);

  std::map<MessageId, cabana::Msg> getMessages(uint8_t source);
  const cabana::Msg *msg(const MessageId &id) const;
  const cabana::Msg *msg(uint8_t source, const QString &name);

  QStringList signalNames() const;
  int signalCount(const MessageId &id) const;
  int msgCount() const;
  int dbcCount(bool no_empty = false) const;
  inline int nonEmptyDBCCount() const { return dbcCount(false); }
  SourceSet sources(DBCFile *file) const;

  const std::set<std::shared_ptr<DBCFile>> allDBCFiles() const;
  const std::vector<std::shared_ptr<DBCFile>> &findDBCFiles(const uint8_t source) const;
  DBCFile *findDBCFile(const MessageId &id) const;

 signals:
  void signalAdded(MessageId id, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id);
  void DBCFileChanged();

private:
  std::map<int, std::vector<std::shared_ptr<DBCFile>>> dbc_files;
};

DBCManager *dbc();

inline QString msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}

inline QString toString(SourceSet ss) {
  if (ss == SOURCE_ALL) {
    return "all";
  } else {
    QStringList ret;
    QList source_list = ss.values();
    std::sort(source_list.begin(), source_list.end());
    for (auto s : source_list) {
      ret << QString::number(s);
    }
    return ret.join(", ");
  }
}
