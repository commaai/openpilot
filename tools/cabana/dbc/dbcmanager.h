#pragma once

#include <map>
#include <optional>

#include <QList>
#include <QMetaType>
#include <QObject>
#include <QString>
#include <QSet>

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcfile.h"

typedef QSet<uint8_t> SourceSet;
const SourceSet SOURCE_ALL = {};
const int INVALID_SOURCE = 0xff;

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent) {}
  ~DBCManager() {}
  bool open(SourceSet s, const QString &dbc_file_name, QString *error = nullptr);
  bool open(SourceSet s, const QString &name, const QString &content, QString *error = nullptr);
  void close(SourceSet s);
  void close(DBCFile *dbc_file);
  void closeAll();
  void removeSourcesFromFile(DBCFile *dbc_file, SourceSet s);

  void addSignal(const MessageId &id, const cabana::Signal &sig);
  void updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig);
  void removeSignal(const MessageId &id, const QString &sig_name);

  void updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment);
  void removeMsg(const MessageId &id);

  QString newMsgName(const MessageId &id);
  QString newSignalName(const MessageId &id);

  const QList<uint8_t>& mask(const MessageId &id) const;

  std::map<MessageId, cabana::Msg> getMessages(uint8_t source);
  const cabana::Msg *msg(const MessageId &id) const;
  const cabana::Msg* msg(uint8_t source, const QString &name);

  QStringList signalNames() const;
  int signalCount(const MessageId &id) const;
  int signalCount() const;
  int msgCount() const;
  int dbcCount() const;
  int nonEmptyDBCCount() const;

  std::optional<std::pair<SourceSet, DBCFile*>> findDBCFile(const uint8_t source) const;
  std::optional<std::pair<SourceSet, DBCFile*>> findDBCFile(const MessageId &id) const;

  QList<std::pair<SourceSet, DBCFile*>> dbc_files;

private:
  SourceSet sources;
  QList<uint8_t> empty_mask;

public slots:
  void updateSources(const SourceSet &s);

signals:
  void signalAdded(MessageId id, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id);
  void DBCFileChanged();
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
