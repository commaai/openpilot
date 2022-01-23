#include "selfdrive/ui/replay/consoleui.h"

#include <iostream>

using namespace std::placeholders;

void ConsoleUI::replayMessageOutput(ReplyMsgType type, const char *msg) {
  if (log_window) {
    wattron(log_window, COLOR_PAIR(type));
    wprintw(log_window, "%s\n", msg);
    wattroff(log_window, COLOR_PAIR(type));
    wrefresh(log_window);
  }
}

void ConsoleUI::downloadProgressHandler(uint64_t cur, uint64_t total) {
  if (progress_bar_window) {
    const int width = 70;
    const float progress = cur / (double)total;
    const int pos = width * progress;
    wclear(progress_bar_window);
    wborder(progress_bar_window, ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ');
    std::string s = util::string_format("Downloading [%s>%s]  %d%% %s", std::string(pos, '=').c_str(),
                                        std::string(width - pos, ' ').c_str(), int(progress * 100.0),
                                        formattedDataSize(total).c_str());
    waddstr(progress_bar_window, s.c_str());
    if (cur >= total) {
      wclear(progress_bar_window);
    }
    wrefresh(progress_bar_window);
  }
}

ConsoleUI::ConsoleUI(Replay *replay, QObject *parent) : replay(replay), QObject(parent) {
  installMessageHandler(std::bind(&ConsoleUI::replayMessageOutput, this, _1, _2));
  installDownloadProgressHandler(std::bind(&ConsoleUI::downloadProgressHandler, this, _1, _2));

  system("clear");
  main_window = initscr();

  start_color();
  init_pair((int)ReplyMsgType::Info, COLOR_WHITE, COLOR_BLACK);
  init_pair((int)ReplyMsgType::Debug, 8, COLOR_BLACK);
  init_pair((int)ReplyMsgType::Warning, COLOR_YELLOW, COLOR_BLACK);
  init_pair((int)ReplyMsgType::Critical, COLOR_RED, COLOR_BLACK);

  clear();
  cbreak();
  noecho();
  // printw("Route %s\n", qPrintable(route));
  int height, width;
  getmaxyx(stdscr, height, width);
  progress_bar_window = newwin(3, 150, 3, 1);
  log_window = newwin(height - 2, width - 2, 5, 1);
  scrollok(log_window, true);

  refresh();
  keypad(main_window, true);
  nodelay(main_window, true);  // non-blocking getchar()

  connect(&m_notifier, SIGNAL(activated(int)), SLOT(readyRead()));
  readyRead();
  m_timer.start(1000, this);
}

ConsoleUI::~ConsoleUI() {
  clrtoeol();
  refresh();
  endwin();
}

void ConsoleUI::timerEvent(QTimerEvent *ev) {
  if (ev->timerId() != m_timer.timerId()) return;
  refresh();
}

void ConsoleUI::handle_key(char c) {
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

void ConsoleUI::readyRead() {
  int c;
  while ((c = getch()) != ERR) {
    handle_key(c);
  }
}
