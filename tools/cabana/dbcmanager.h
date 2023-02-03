#pragma once

#include <map>
#include <QObject>
#include <QString>
#include "opendbc/can/common_dbc.h"

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
  void open(const QString &name, const QString &content);
  QString generateDBC();
  void addSignal(const QString &id, const Signal &sig);
  void updateSignal(const QString &id, const QString &sig_name, const Signal &sig);
  void removeSignal(const QString &id, const QString &sig_name);

  static std::pair<uint8_t, uint32_t> parseId(const QString &id);
  inline static std::vector<std::string> allDBCNames() { return get_dbc_names(); }
  inline QString name() const { return dbc ? dbc->name.c_str() : ""; }
  void updateMsg(const QString &id, const QString &name, uint32_t size);
  void removeMsg(const QString &id);
  inline const std::map<uint32_t, DBCMsg> &messages() const { return msgs; }
  inline const DBCMsg *msg(const QString &id) const { return msg(parseId(id).second); }
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

// TODO: Add helper function in dbc.h
double get_raw_value(uint8_t *data, size_t data_size, const Signal &sig);
bool operator==(const Signal &l, const Signal &r);
inline bool operator!=(const Signal &l, const Signal &r) { return !(l == r); }
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
void updateSigSizeParamsFromRange(Signal &s, int start_bit, int size);
std::pair<int, int> getSignalRange(const Signal *s);
DBCManager *dbc();
inline QString msgName(const QString &id, const char *def = "untitled") {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : def;
}
