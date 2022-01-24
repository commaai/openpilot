#pragma once

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
  void handleKey(char c);
  void displayHelp();
  void updateTimeline(int cur_sec, int total_sec);
  void updateStats();
  void updateSubmaster();

  enum Win { Title, Stats, Log, LogBorder, DownloadBar, Timeline, TimelineDesc, Help, CarState, Max};
  WINDOW* w[Win::Max] = {};
  SubMaster sm;
  Replay *replay;
  QBasicTimer getch_timer;
  QTimer sm_timer;
  QSocketNotifier notifier{0, QSocketNotifier::Read, this};

signals:
  void updateProgressBarSignal(uint64_t cur, uint64_t total, bool success);
  void logMessageSignal(int type, const QString &msg);

private slots:
  void readyRead();
  void timerEvent(QTimerEvent *ev);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);
  void logMessage(int type, const QString &msg);
};
