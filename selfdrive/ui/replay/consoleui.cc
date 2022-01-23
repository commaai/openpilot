#include "selfdrive/ui/replay/consoleui.h"

#include <initializer_list>
#include <iostream>
#include <QApplication>

#include "selfdrive/common/params.h"
#include "selfdrive/common/version.h"

using namespace std::placeholders;
enum class Color {
  Info,
  Debug,
  Warning,
  Critical,
  bgTimeLine,
  bgTitle,
  Played,
  Engaged,
  EngagedPlayed,
  CarEvent,
  CarEventPlayed,
  AlertWarning,
  AlertWarningPlayed,

};

ConsoleUI::ConsoleUI(Replay *replay, QObject *parent) : replay(replay), sm({"carState", "liveParameters"}), QObject(parent) {
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
  init_pair((int)Color::Engaged, COLOR_BLUE, COLOR_BLUE);
  init_pair((int)Color::EngagedPlayed, 22, 22);
  init_pair((int)Color::CarEvent, 10, 10);
  init_pair((int)Color::CarEventPlayed, COLOR_YELLOW, COLOR_YELLOW);
  init_pair((int)Color::AlertWarning, COLOR_RED, COLOR_RED);

  int height, width;
  getmaxyx(stdscr, height, width);

  title = newwin(1, width , 0, 0);
  stats_win = newwin(2, width, 2, 3);

  timeline_win = newwin(2, 100, 5, 3);
  wbkgd(timeline_win, COLOR_PAIR(Color::bgTimeLine));
  timeline_desc_win = newwin(1, 100, 8, 3);

  progress_bar_window = newwin(1, 60, 10, 3);
  car_state_win = newwin(5, 100, 11, 3);
  
  log_window = newwin(height - 30, 100, 17, 3);
  scrollok(log_window, true);

  help_win = newwin(8, 50, height-10, 3);
  

  refresh();
  keypad(main_window, true);
  nodelay(main_window, true);  // non-blocking getchar()

  displayHelp();

  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateTimeline);
  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateStats);
  QObject::connect(&m_notifier, SIGNAL(activated(int)), SLOT(readyRead()));
  readyRead();
  m_timer.start(1000, this);

  sm_timer.callOnTimeout(this, &ConsoleUI::update);
  sm_timer.start(50);
  

  wbkgd(title, COLOR_PAIR(Color::bgTitle));
  mvwprintw(title, 0, 3, "openpilot replay %s", COMMA_VERSION);
  wrefresh(title);

  std::pair<Color, const char *> indicators[] {
    {Color::Engaged, " engaged "},
    {Color::CarEvent, " alert "},
    {Color::AlertWarning, " warning "},
  };
  for (auto [color, name] : indicators) {
    wattron(timeline_desc_win, COLOR_PAIR(color));
    waddstr(timeline_desc_win, "  ");
    wattroff(timeline_desc_win, COLOR_PAIR(color));
    waddstr(timeline_desc_win, name);
  }
  wrefresh(timeline_desc_win);
  wrefresh(timeline_win);
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

void ConsoleUI::update() {
  sm.update(0);
  if (sm.updated("carState")) {
    mvwprintw(car_state_win, 0, 0, "SPEED: %.2f m/s", sm["carState"].getCarState().getVEgo());
  }
  if (sm.updated("liveParameters")) {
    auto p = sm["liveParameters"].getLiveParameters();
    mvwprintw(car_state_win, 1, 0, "ANGLE OFFSET (AVG): %.2f deg", p.getAngleOffsetAverageDeg());
    mvwprintw(car_state_win, 2, 0, "ANGLE OFFSET (INSTANT): %.2f deg", p.getAngleOffsetDeg());
    mvwprintw(car_state_win, 3, 0, "STIFFNESS: %.2f %%", p.getStiffnessFactor() * 100);
    mvwprintw(car_state_win, 4, 0, "STEER RATIO: %.2f", p.getSteerRatio());
  }
  wrefresh(car_state_win);
}

void ConsoleUI::replayMessageOutput(ReplyMsgType type, const char *msg) {
  if (log_window) {
    wattron(log_window, COLOR_PAIR(type));
    wprintw(log_window, "%s\n", msg);
    wattroff(log_window, COLOR_PAIR(type));
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
    {"e", "next engagement"},
    {"d", "next disengagement"},
    {"x", "play at full speed"},
    {"q", "exit"},
    {"enter", "enter seek request"},
    {"shift + g", "play from start"},
  };
  
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

void ConsoleUI::updateStats(int cur_sec, int total_sec) {
  mvwprintw(stats_win, 0, 0, "Route: %s", qPrintable(replay->route()->name()));
  mvwprintw(stats_win, 1, 0, "Current: %d s  Total %d s         ", cur_sec, total_sec);
  wrefresh(stats_win);
}

void ConsoleUI::updateTimeline(int cur_sec, int total_sec) {
  auto draw_at = [=](int x, char c = ' ') {
    mvwaddch(timeline_win, 0, x, c);
    mvwaddch(timeline_win, 1, x, c);
  };
  auto remove_at = [=](int x) {
    mvwdelch(timeline_win, 0, x);
    mvwdelch(timeline_win, 1, x);
  };

  int width = getmaxx(timeline_win);
  int cur_pos = ((double)cur_sec / total_sec) * width;
  wattron(timeline_win, COLOR_PAIR(Color::Played));
  for (int i = 0; i < width; ++i) {
    remove_at(i);
    if (i <= cur_pos) {
      draw_at(i, ' ');
    }
  }
  wattroff(timeline_win, COLOR_PAIR(Color::Played));
  
  auto summary = replay->getSummary();
  for (auto [engage_sec, disengage_sec] : summary) {
    int start_pos = ((double)engage_sec/total_sec) * width;
    int end_pos = ((double)disengage_sec/total_sec) * width;
    for (int i = start_pos; i < end_pos; ++i) {
      wattron(timeline_win, COLOR_PAIR(i < cur_pos ? Color::EngagedPlayed : Color::Engaged));
      remove_at(i);
      draw_at(i, ' ');
    }
  }

  auto car_events = replay->getCarEvents();
  for (auto [start_sec, end_sec, status] : car_events) {
    int start_pos = ((double)start_sec / total_sec) * width;
    int end_pos = ((double)end_sec / total_sec) * width;
    for (int i = start_pos; i < end_pos; ++i) {
      wattron(timeline_win, COLOR_PAIR(i < cur_pos ? Color::CarEventPlayed : Color::CarEvent));
      remove_at(i);
      draw_at(i, ' ');
    }
  }

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
    case 'g':
      replay->seekTo(0, false);
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
