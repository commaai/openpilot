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
  void handle_key(char c);
  void replayMessageOutput(ReplyMsgType type, const char *msg);
  void downloadProgressHandler(uint64_t cur, uint64_t total);
  void displayHelp();
  void updateTimeline(int cur_sec, int total_sec);
  void updateStats(int cur_sec, int total_sec);
  void update();

  QSocketNotifier m_notifier{0, QSocketNotifier::Read, this};
  QBasicTimer m_timer;
  QTimer sm_timer;
  Replay *replay;
  WINDOW *title;
  WINDOW *main_window;
  WINDOW *stats_win;
  WINDOW *log_window;
  WINDOW *progress_bar_window;
  WINDOW *timeline_win;
  WINDOW *help_win;
  WINDOW *car_state_win;
  WINDOW *timeline_desc_win;
  WINDOW *seek_win;
  SubMaster sm;

 private slots:
  void readyRead();
  void timerEvent(QTimerEvent *ev);
};
