#pragma once

#include <atomic>
#include <map>

#include <QApplication>
#include <QHash>
#include <QObject>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/replay/replay.h"

const int FPS = 20;
const static int LOG_SIZE = 25;

struct CanData {
  QString id;
  double ts;
  uint32_t address;
  uint16_t bus_time;
  uint8_t source;
  QByteArray dat;
};

class Parser : public QObject {
  Q_OBJECT

public:
  Parser(QObject *parent);
  ~Parser();
  static uint32_t addressFromId(const QString &id);
  bool eventFilter(const Event *event);
  bool loadRoute(const QString &route, const QString &data_dir, bool use_qcam);
  void openDBC(const QString &name);
  void saveDBC(const QString &name) {}
  void addNewMsg(const Msg &msg);
  void removeSignal(const QString &id, const QString &sig_name);
  void seekTo(double ts);
  const Signal *getSig(const QString &id, const QString &sig_name);
  void setRange(double min, double max);
  void resetRange();
  void setCurrentMsg(const QString &id);
  inline std::pair<double, double> range() const { return {begin_sec, end_sec}; }
  inline double currentSec() const { return current_sec; }
  inline bool isZoomed() const { return is_zoomed; }
  inline const Msg *getMsg(const QString &id) { return getMsg(addressFromId(id)); }
  inline const Msg *getMsg(uint32_t address) {
    auto it = msg_map.find(address);
    return it != msg_map.end() ? it->second : nullptr;
  }

signals:
  void showPlot(const QString &id, const QString &name);
  void hidePlot(const QString &id, const QString &name);
  void signalRemoved(const QString &id, const QString &sig_name);
  void eventsMerged();
  void rangeChanged(double min, double max);
  void received(std::vector<CanData> can);
  void updated();

public:
  Replay *replay = nullptr;
  QHash<QString, uint64_t> counters;
  std::map<QString, CanData> can_msgs;
  QList<CanData> history_log;

protected:
  void process(std::vector<CanData> can);
  void segmentsMerged();

  std::atomic<double> current_sec = 0.;
  std::atomic<bool> seeking = false;
  QString dbc_name;
  double begin_sec = 0;
  double end_sec = 0;
  double event_begin_sec = 0;
  double event_end_sec = 0;
  bool is_zoomed = false;
  DBC *dbc = nullptr;
  std::map<uint32_t, const Msg *> msg_map;
  QString current_msg_id;
  std::vector<CanData> msgs_buf;
};

Q_DECLARE_METATYPE(std::vector<CanData>);

// TODO: Add helper function in dbc.h
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
inline QString toHex(const QByteArray &dat) {
  return dat.toHex(' ').toUpper();
}

extern Parser *parser;
