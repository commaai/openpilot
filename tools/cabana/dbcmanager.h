#pragma once

#include <QObject>

#include "opendbc/can/common_dbc.h"

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
  inline const DBC *getDBC() const { return dbc; }
  inline const Msg *msg(const QString &id) const { return msg(parseId(id).second); }
  inline const Msg *msg(uint32_t address) const {
    auto it = msg_map.find(address);
    return it != msg_map.end() ? it->second : nullptr;
  }

signals:
  void signalAdded(const Signal *sig);
  void signalRemoved(const Signal *sig);
  void signalUpdated(const Signal *sig);
  void msgUpdated(uint32_t address);
  void msgRemoved(uint32_t address);
  void DBCFileChanged();

private:
  void updateMsgMap();
  DBC *dbc = nullptr;
  std::unordered_map<uint32_t, const Msg *> msg_map;
};

// TODO: Add helper function in dbc.h
double get_raw_value(uint8_t *data, size_t data_size, const Signal &sig);
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
void updateSigSizeParamsFromRange(Signal &s, int from, int to);
std::pair<int, int> getSignalRange(const Signal *s);
DBCManager *dbc();
inline QString msgName(const QString &id, const char *def = "untitled") {
  auto msg = dbc()->msg(id);
  return msg ? msg->name.c_str() : def;
}
