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

  enum Win { Title, Stats, Log, DownloadBar, Timeline, TimelineDesc, Help, CarState, Seek };
  WINDOW* w[Win::Seek + 1] = {};
  SubMaster sm;
  Replay *replay;
  QBasicTimer getch_timer;
  QTimer sm_timer;
  QSocketNotifier m_notifier{0, QSocketNotifier::Read, this};

 private slots:
  void readyRead();
  void timerEvent(QTimerEvent *ev);
};
