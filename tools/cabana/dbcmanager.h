#pragma once

#include <map>
#include <QList>
#include <QMetaType>
#include <QObject>
#include <QString>

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

uint qHash(const MessageId &item);
Q_DECLARE_METATYPE(MessageId);

namespace dbcmanager {

typedef QList<std::pair<QString, QString>> ValueDescription;

struct Signal {
  QString name;
  int start_bit, msb, lsb, size;
  bool is_signed;
  double factor, offset;
  bool is_little_endian;
  QString min, max, unit;
  QString comment;
  ValueDescription val_desc;
};

struct Msg {
  QString name;
  uint32_t size;

  std::vector<const Signal*> getSignals() const;
  const Signal *sig(const QString &sig_name) const {
    auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.name == sig_name; });
    return it != sigs.end() ? &(*it) : nullptr;
  }

private:
  QList<Signal> sigs;
  friend class DBCManager;
};

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent) {}
  ~DBCManager() {}
  bool open(const QString &dbc_file_name, QString *error = nullptr);
  bool open(const QString &name, const QString &content, QString *error = nullptr);
  QString generateDBC();
  void addSignal(const MessageId &id, const Signal &sig);
  void updateSignal(const MessageId &id, const QString &sig_name, const Signal &sig);
  void removeSignal(const MessageId &id, const QString &sig_name);

  static std::vector<std::string> allDBCNames();
  inline QString name() const { return name_; }
  void updateMsg(const MessageId &id, const QString &name, uint32_t size);
  void removeMsg(const MessageId &id);
  inline const std::map<uint32_t, Msg> &messages() const { return msgs; }
  inline const Msg *msg(const MessageId &id) const { return msg(id.address); }
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

}  // namespace dbcmanager
