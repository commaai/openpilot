#include "selfdrive/ui/replay/consoleui.h"

#include <initializer_list>
#include <iostream>
#include <QApplication>

using namespace std::placeholders;
enum class Color {
  Info,
  Debug,
  Warning,
  Critical,
  bgTimeLine,
  bgTitle,
  Played,
};

ConsoleUI::ConsoleUI(Replay *replay, QObject *parent) : replay(replay), QObject(parent) {
  installMessageHandler(std::bind(&ConsoleUI::replayMessageOutput, this, _1, _2));
  installDownloadProgressHandler(std::bind(&ConsoleUI::downloadProgressHandler, this, _1, _2));

  system("clear");
  main_window = initscr();
  curs_set(0);
  start_color();
  clear();
  cbreak();
  noecho();

  // Initialize all the colors
  init_pair((int)Color::Info, COLOR_WHITE, COLOR_BLACK);
  init_pair((int)Color::Debug, 8, COLOR_BLACK);
  init_pair((int)Color::Warning, COLOR_YELLOW, COLOR_BLACK);
  init_pair((int)Color::Critical, COLOR_RED, COLOR_BLACK);
  init_pair((int)Color::bgTimeLine, COLOR_BLACK, 8);
  init_pair((int)Color::bgTitle, COLOR_BLACK, COLOR_WHITE);
  init_pair((int)Color::Played, COLOR_GREEN, COLOR_GREEN);

  int height, width;
  getmaxyx(stdscr, height, width);

  title = newwin(1, width , 0, 0);
  timeline_win = newwin(3, 100, 4, 3);
  wbkgd(timeline_win, COLOR_PAIR(Color::bgTimeLine));

  progress_bar_window = newwin(3, 60, 12, 3);
  log_window = newwin(height - 25, 100, 15, 3);
  box(log_window, 0, 0);
  scrollok(log_window, true);
  // wrefresh(log_window);

  help_win = newwin(7, 50, height-7, 3);
  

  refresh();
  keypad(main_window, true);
  nodelay(main_window, true);  // non-blocking getchar()

  // displayTimeline();
  displayHelp();

  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateTimeline);
  connect(&m_notifier, SIGNAL(activated(int)), SLOT(readyRead()));
  readyRead();
  m_timer.start(1000, this);

  wbkgd(title, COLOR_PAIR(Color::bgTitle));
  mvwprintw(title, 0, 3, "Replaying %s", qPrintable(replay->route()->name()));
  wrefresh(title);
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

void ConsoleUI::replayMessageOutput(ReplyMsgType type, const char *msg) {
  // static int y = 1;
  if (log_window) {
    // box(log_window, 0, 0);
    wattron(log_window, COLOR_PAIR(type));
    wprintw(log_window, "%s\n", msg);
    wattroff(log_window, COLOR_PAIR(type));
    // scroll(log_window);
    wrefresh(log_window);
  }
}

void ConsoleUI::displayHelp() {
  std::initializer_list<std::pair<const char *, const char*>> single_line_keys {
    {"s", "+10s"},
    {"shift+s", "-10s"},
    {"m", "+60s"},
    {"shift+m", "+60s"},
  };

  std::initializer_list<std::pair<const char *, const char*>> multi_line_keys = {
    {"f", "full speed"},
    {"e", "next engagement"},
    {"d", "next disengagements"},
    {"q", "exit"}
  };
  
  wclear(help_win);
  auto write_shortcut = [=](std::string key, const char *desc) {
    wattron(help_win, COLOR_PAIR(Color::bgTitle));
    waddstr(help_win, (' ' + key + ' ').c_str());
    wattroff(help_win, COLOR_PAIR(Color::bgTitle));
    waddstr(help_win, " ");
    waddstr(help_win, desc);
    waddstr(help_win, "  ");
  };

  for (auto [key, desc] : single_line_keys) {
    write_shortcut(key, desc);
  }
  int y = 2;
  for (auto [key, desc] : multi_line_keys) {
    wmove(help_win, y++, 0);
    write_shortcut(key, desc);
  }
  wrefresh(help_win);
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

void ConsoleUI::updateTimeline(int cur_sec, int total_sec) {
  // wclear(timeline_win);
  // std::vector<std::tuple<uint32_t, uint32_t>> timelines;
  std::string bar(200, ' ');
  // for (auto [start, end] : timelines) {
  int width = getmaxx(timeline_win);
  int cur_pos = ((double)cur_sec / total_sec) * width;
  wattron(timeline_win, COLOR_PAIR(Color::Played));
  for (int i = 0; i < cur_pos; ++i) {
    mvwaddch(timeline_win, 0, i, ' ');
    mvwaddch(timeline_win, 1, i, ' ');
    mvwaddch(timeline_win, 2, i, ' ');
  }
  for (int i = cur_pos; i < width; ++i) {
    mvwdelch(timeline_win, 0, i);
    mvwdelch(timeline_win, 1, i);
    mvwdelch(timeline_win, 2, i);
  }
  wattroff(timeline_win, COLOR_PAIR(Color::Played));
  wrefresh(timeline_win);
}

void ConsoleUI::readyRead() {
  int c;
  while ((c = getch()) != ERR) {
    handle_key(c);
  }
}

void ConsoleUI::handle_key(char c) {
  switch (c) {
    case '\n': {
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
      break;
    }
      // getch2();  // remove \n from entering seek

    case 'e':
      replay->seekToFlag(FindFlag::nextEngagement);
      break;
    case 'd':
      replay->seekToFlag(FindFlag::nextDisEngagement);
      break;
    case 'm':
      replay->seekTo(+60, true);
      break;
    case 'M':
      replay->seekTo(-60, true);
      break;
    case 's':
      replay->seekTo(+10, true);
      break;
    case 'S':
      replay->seekTo(-10, true);
      break;
    case 'G':
      replay->seekTo(0, true);
      break;
    case 'x':
      if (replay->hasFlag(REPLAY_FLAG_FULL_SPEED)) {
        replay->removeFlag(REPLAY_FLAG_FULL_SPEED);
        qInfo() << "replay at normal speed";
      } else {
        replay->addFlag(REPLAY_FLAG_FULL_SPEED);
        qInfo() << "replay at full speed";
      }
      break;
    case ' ':
      replay->pause(!replay->isPaused());
      break;
    case 'q':
    case 'Q':
      qApp->exit();
      break;
  }
}
