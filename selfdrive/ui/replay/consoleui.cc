#include "selfdrive/ui/replay/consoleui.h"

#include <ncurses.h>

#include <QApplication>
#include <iostream>

Keyboard::Keyboard(Replay *replay, QObject *parent) : replay(replay), QObject(parent) {
  connect(&m_notifier, SIGNAL(activated(int)), SLOT(readyRead()));
  readyRead();  // data might be already available without notification
  m_timer.start(1000, this);
}

void Keyboard::timerEvent(QTimerEvent *ev) {
  if (ev->timerId() != m_timer.timerId()) return;
  refresh();
}

void Keyboard::handle_key(char c) {
  if (c == '\n') {
    printf("Enter seek request: ");
    std::string r;
    std::cin >> r;

    try {
      if (r[0] == '#') {
        r.erase(0, 1);
        replay->seekTo(std::stoi(r) * 60, false);
      } else {
        replay->seekTo(std::stoi(r), false);
      }
    } catch (std::invalid_argument) {
      qDebug() << "invalid argument";
    }
    // getch2();  // remove \n from entering seek
  } else if (c == 'e') {
    replay->seekToFlag(FindFlag::nextEngagement);
  } else if (c == 'd') {
    replay->seekToFlag(FindFlag::nextDisEngagement);
  } else if (c == 'm') {
    replay->seekTo(+60, true);
  } else if (c == 'M') {
    replay->seekTo(-60, true);
  } else if (c == 's') {
    replay->seekTo(+10, true);
  } else if (c == 'S') {
    replay->seekTo(-10, true);
  } else if (c == 'G') {
    replay->seekTo(0, true);
  } else if (c == 'x') {
    if (replay->hasFlag(REPLAY_FLAG_FULL_SPEED)) {
      replay->removeFlag(REPLAY_FLAG_FULL_SPEED);
      qInfo() << "replay at normal speed";
    } else {
      replay->addFlag(REPLAY_FLAG_FULL_SPEED);
      qInfo() << "replay at full speed";
    }
  } else if (c == ' ') {
    replay->pause(!replay->isPaused());
  }
}

void Keyboard::readyRead() {
  int c;
  while ((c = getch()) != ERR) {
    handle_key(c);
  }
}
