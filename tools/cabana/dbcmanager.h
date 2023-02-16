#pragma once

#include <map>
#include <QObject>
#include <QString>
#include "opendbc/can/common_dbc.h"

struct MessageId {
  uint8_t source;
  uint32_t address;

  QString toString() const {
    return QString("%1:%2").arg(source).arg(address, 1, 16);
  }

  bool operator==(const MessageId &other) const {
    return source == other.source && address == other.address;
  }

  bool operator!=(const MessageId &other) const {
    return !(*this == other);
  }

  bool operator<(const MessageId &other) const {
    return std::pair{source, address} < std::pair{other.source, other.address};
  }

  bool operator>(const MessageId &other) const {
    return std::pair{source, address} > std::pair{other.source, other.address};
  }
};

Q_DECLARE_METATYPE(MessageId);

uint qHash(const MessageId &item);

struct DBCMsg {
  QString name;
  uint32_t size;
  // signal must be saved as value in map to make undo stack work properly.
  std::map<QString, Signal> sigs;
  // return vector<signals>, sort by start_bits
  std::vector<const Signal*> getSignals() const;
};

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent);
  ~DBCManager();

  void open(const QString &dbc_file_name);
  bool open(const QString &name, const QString &content, QString *error = nullptr);
  QString generateDBC();
  void addSignal(const MessageId &id, const Signal &sig);
  void updateSignal(const MessageId &id, const QString &sig_name, const Signal &sig);
  void removeSignal(const MessageId &id, const QString &sig_name);

  inline static std::vector<std::string> allDBCNames() { return get_dbc_names(); }
  inline QString name() const { return dbc ? dbc->name.c_str() : ""; }
  void updateMsg(const MessageId &id, const QString &name, uint32_t size);
  void removeMsg(const MessageId &id);
  inline const std::map<uint32_t, DBCMsg> &messages() const { return msgs; }
  inline const DBCMsg *msg(const MessageId &id) const { return msg(id.address); }
  inline const DBCMsg *msg(uint32_t address) const {
    auto it = msgs.find(address);
    return it != msgs.end() ? &it->second : nullptr;
  }

signals:
  void signalAdded(uint32_t address, const Signal *sig);
  void signalRemoved(const Signal *sig);
  void signalUpdated(const Signal *sig);
  void msgUpdated(uint32_t address);
  void msgRemoved(uint32_t address);
  void DBCFileChanged();

private:
  void initMsgMap();
  DBC *dbc = nullptr;
  std::map<uint32_t, DBCMsg> msgs;
};

const QString UNTITLED = "untitled";

// TODO: Add helper function in dbc.h
double get_raw_value(uint8_t *data, size_t data_size, const Signal &sig);
bool operator==(const Signal &l, const Signal &r);
inline bool operator!=(const Signal &l, const Signal &r) { return !(l == r); }
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
void updateSigSizeParamsFromRange(Signal &s, int start_bit, int size);
std::pair<int, int> getSignalRange(const Signal *s);
DBCManager *dbc();
inline QString msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}
