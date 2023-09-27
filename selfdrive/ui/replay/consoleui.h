#pragma once

#include <array>
#include <QBasicTimer>
#include <QObject>
#include <QSocketNotifier>
#include <QTimer>
#include <QTimerEvent>

#include "selfdrive/ui/replay/replay.h"
#include <ncurses.h>

class ConsoleUI : public QObject {
  Q_OBJECT

public:
  ConsoleUI(Replay *replay, QObject *parent = 0);
  ~ConsoleUI();

private:
  void initWindows();
  void handleKey(char c);
  void displayHelp();
  void displayTimelineDesc();
  void updateTimeline();
  void updateSummary();
  void updateStatus();
  void pauseReplay(bool pause);

  enum Status { Waiting, Playing, Paused };
  enum Win { Title, Stats, Log, LogBorder, DownloadBar, Timeline, TimelineDesc, Help, CarState, Max};
  std::array<WINDOW*, Win::Max> w{};
  SubMaster sm;
  Replay *replay;
  QBasicTimer getch_timer;
  QTimer sm_timer;
  QSocketNotifier notifier{0, QSocketNotifier::Read, this};
  int max_width, max_height;
  Status status = Status::Waiting;

signals:
  void updateProgressBarSignal(uint64_t cur, uint64_t total, bool success);
  void logMessageSignal(ReplyMsgType type, const QString &msg);

private slots:
  void readyRead();
  void timerEvent(QTimerEvent *ev);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);
  void logMessage(ReplyMsgType type, const QString &msg);
};
