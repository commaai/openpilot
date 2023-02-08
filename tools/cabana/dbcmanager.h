#pragma once

#include <map>
#include <QList>
#include <QString>

namespace dbcmanager {

struct Signal {
  QString name;
  int start_bit, msb, lsb, size;
  bool is_signed;
  double factor, offset;
  bool is_little_endian;
  QString min, max, unit;
  QString comment, val_desc;
};

struct Msg {
  QString name;
  uint32_t size;
  QList<Signal> sigs;

  const Signal *sig(const QString &sig_name) const {
    auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.name == sig_name; });
    return it != sigs.end() ? &(*it) : nullptr;
  }
};

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent) {}
  ~DBCManager() {}
  void open(const QString &dbc_file_name);
  void open(const QString &name, const QString &content);
  QString generateDBC();
  void addSignal(const QString &id, const Signal &sig);
  void updateSignal(const QString &id, const QString &sig_name, const Signal &sig);
  void removeSignal(const QString &id, const QString &sig_name);
  static std::pair<uint8_t, uint32_t> parseId(const QString &id);
  static std::vector<std::string> allDBCNames();
  inline QString name() const { return name_; }
  void updateMsg(const QString &id, const QString &name, uint32_t size);
  void removeMsg(const QString &id);
  inline const std::map<uint32_t, Msg> &messages() const { return msgs; }
  inline const Msg *msg(const QString &id) const { return msg(parseId(id).second); }
  inline const Msg *msg(uint32_t address) const {
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
  void parseExtraInfo(const QString &content);
  std::map<uint32_t, Msg> msgs;
  QString name_;
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

}  // namespace dbcmanager
