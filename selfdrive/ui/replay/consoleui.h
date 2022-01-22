#pragma once

#include <QBasicTimer>
#include <QObject>
#include <QSocketNotifier>
#include <QTimerEvent>

#include "selfdrive/ui/replay/replay.h"

class Keyboard : public QObject {
  Q_OBJECT

 public:
  Keyboard(Replay *replay, QObject *parent = 0);

 private:
  void handle_key(char c);

  QSocketNotifier m_notifier{0, QSocketNotifier::Read, this};
  QBasicTimer m_timer;
  Replay *replay;

  private slots:
  void readyRead();
  void timerEvent(QTimerEvent *ev);
};
