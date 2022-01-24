#include "selfdrive/ui/replay/consoleui.h"

#include <initializer_list>
#include <iostream>
#include <QApplication>

#include "selfdrive/common/params.h"
#include "selfdrive/common/version.h"

using namespace std::placeholders;
enum Color {
  Info,
  Debug,
  Warning,
  Critical,
  bgTitle,
  Engaged,
  Disengaged,
  Alert,
  AlertWarning,
  PosIndicator,
};

ConsoleUI::ConsoleUI(Replay *replay, QObject *parent) : replay(replay), sm({"carState", "liveParameters"}), QObject(parent) {
  qRegisterMetaType<uint64_t>("uint64_t");
  installMessageHandler(std::bind(&ConsoleUI::logMessageHandler, this, _1, _2));
  installDownloadProgressHandler(std::bind(&ConsoleUI::downloadProgressHandler, this, _1, _2));

  system("clear");
  initscr();
  curs_set(0);
  start_color();
  clear();
  cbreak();
  noecho();
  keypad(stdscr, true);
  nodelay(stdscr, true);  // non-blocking getchar()

  // Initialize all the colors
  init_pair(Color::Info, COLOR_WHITE, COLOR_BLACK);
  init_pair(Color::Debug, 8, COLOR_BLACK);
  init_pair(Color::Warning, COLOR_YELLOW, COLOR_BLACK);
  init_pair(Color::Critical, COLOR_RED, COLOR_BLACK);
  init_pair(Color::Disengaged, COLOR_BLUE, COLOR_BLUE);
  init_pair(Color::bgTitle, COLOR_BLACK, COLOR_WHITE);
  init_pair(Color::Engaged, 28, 28);
  init_pair(Color::Alert, 11, 11);
  init_pair(Color::AlertWarning, COLOR_RED, COLOR_RED);
  init_pair(Color::PosIndicator, 15, COLOR_BLACK);
  

  int height, width;
  getmaxyx(stdscr, height, width);

  w[Win::Title] = newwin(1, width , 0, 0);
  w[Win::Stats] = newwin(2, width, 2, 3);
  w[Win::Timeline] = newwin(4, 100, 5, 3);
  w[Win::TimelineDesc]= newwin(1, 100, 10, 3);
  w[Win::DownloadBar] = newwin(1, 100, 11, 3);
  w[Win::CarState] = newwin(5, 100, 13, 3);
  if (int log_height = height - 29; log_height > 5) {
    w[Win::LogBorder] = newwin(log_height, 100, 19, 2);
    box(w[Win::LogBorder], 0, 0);
    w[Win::Log] = newwin(log_height - 2, 98, 20, 3);
    scrollok(w[Win::Log], true);
  }
  w[Win::Help] = newwin(5, 100, height-6, 3);
  
  wbkgd(w[Win::Title], COLOR_PAIR(Color::bgTitle));
  
  refresh();

  mvwprintw(w[Win::Title], 0, 3, "openpilot replay %s", COMMA_VERSION);

  std::pair<Color, const char *> indicators[] {
    {Color::Engaged, " Engaged "},
    {Color::Disengaged, " Disengaged "},
    {Color::Alert, " Alert "},
    {Color::AlertWarning, " Warning "},
  };
  for (auto [color, name] : indicators) {
    wattron(w[Win::TimelineDesc], COLOR_PAIR(color));
    waddstr(w[Win::TimelineDesc], "  ");
    wattroff(w[Win::TimelineDesc], COLOR_PAIR(color));
    waddstr(w[Win::TimelineDesc], name);
  }

  wrefresh(w[Win::Title]);
  wrefresh(w[Win::TimelineDesc]);
  wrefresh(w[Win::Timeline]);
  if (w[Win::Log]) {
    wrefresh(w[Win::LogBorder]);
    wrefresh(w[Win::Log]);
  }
  displayHelp();
  updateStats(0, replay->route()->segments().size() * 60);
  updateTimeline(0, replay->route()->segments().size() * 60);

  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateTimeline);
  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateStats);
  QObject::connect(&m_notifier, SIGNAL(activated(int)), SLOT(readyRead()));
  QObject::connect(this, &ConsoleUI::updateProgressBarSignal, this, &ConsoleUI::updateProgressBar);
  QObject::connect(this, &ConsoleUI::logMessageSignal, this, &ConsoleUI::logMessage);

  readyRead();

  getch_timer.start(1000, this);
  sm_timer.callOnTimeout(this, &ConsoleUI::update);
  sm_timer.start(50);
}

ConsoleUI::~ConsoleUI() {
  endwin();
}

void ConsoleUI::timerEvent(QTimerEvent *ev) {
  if (ev->timerId() != getch_timer.timerId()) return;
  refresh();
}

void ConsoleUI::update() {
  sm.update(0);
  if (sm.updated("carState")) {
    mvwprintw(w[Win::CarState], 0, 0, "SPEED: %.2f m/s", sm["carState"].getCarState().getVEgo());
  }
  if (sm.updated("liveParameters")) {
    auto p = sm["liveParameters"].getLiveParameters();
    mvwprintw(w[Win::CarState], 1, 0, "ANGLE OFFSET (AVG): %.2f deg", p.getAngleOffsetAverageDeg());
    mvwprintw(w[Win::CarState], 2, 0, "ANGLE OFFSET (INSTANT): %.2f deg", p.getAngleOffsetDeg());
    mvwprintw(w[Win::CarState], 3, 0, "STIFFNESS: %.2f %%", p.getStiffnessFactor() * 100);
    mvwprintw(w[Win::CarState], 4, 0, "STEER RATIO: %.2f", p.getSteerRatio());
  }
  wrefresh(w[Win::CarState]);
}

void ConsoleUI::displayHelp() {
  std::initializer_list<std::pair<const char *, const char*>> single_line_keys {
    {"s", "+10s"},
    {"shift+s", "-10s"},
    {"m", "+60s"},
    {"shift+m", "+60s"},
    {"p", "Pause/Resume"},
    {"e", "Next Engmt"},
    {"d", "Next DisEngmt"},
  };
  std::initializer_list<std::pair<const char *, const char*>> multi_line_keys = {
    {"enter", "Enter seek request"},
    {"x", "Replay at full speed"},
    {"q", "Exit"},
  };
  
  auto write_shortcut = [=](std::string key, std::string desc) {
    wattron(w[Win::Help], COLOR_PAIR(Color::bgTitle));
    waddstr(w[Win::Help], (' ' + key + ' ').c_str());
    wattroff(w[Win::Help], COLOR_PAIR(Color::bgTitle));
    waddstr(w[Win::Help], (" " + desc + " ").c_str());
  };

  for (auto [key, desc] : single_line_keys) {
    write_shortcut(key, desc);
  }
  int y = 2;
  for (auto [key, desc] : multi_line_keys) {
    wmove(w[Win::Help], y++, 0);
    write_shortcut(key, desc);
  }
  wrefresh(w[Win::Help]);
}

void ConsoleUI::logMessageHandler(ReplyMsgType type, const char *msg) {
  emit logMessageSignal((int)type ,msg);
}

void ConsoleUI::downloadProgressHandler(uint64_t cur, uint64_t total) {
  emit updateProgressBarSignal(cur, total);
}

void ConsoleUI::logMessage(int type, const char *msg) {
  if (w[Win::Log]) {
    wattron(w[Win::Log], COLOR_PAIR((int)type));
    wprintw(w[Win::Log], "%s\n", msg);
    wattroff(w[Win::Log], COLOR_PAIR((int)type));
    wrefresh(w[Win::Log]);
  }
}

void ConsoleUI::updateProgressBar(uint64_t cur, uint64_t total) {
  const int width = 30;
  const float progress = cur / (double)total;
  const int pos = width * progress;
  werase(w[Win::DownloadBar]);
  std::string s = util::string_format("Downloading [%s>%s]  %d%% %s", std::string(pos, '=').c_str(),
                                      std::string(width - pos, ' ').c_str(), int(progress * 100.0),
                                      formattedDataSize(total).c_str());
  waddstr(w[Win::DownloadBar], s.c_str());
  if (cur >= total) {
    werase(w[Win::DownloadBar]);
  }
  wrefresh(w[Win::DownloadBar]);
}

void ConsoleUI::updateStats(int cur_sec, int total_sec) {
  mvwprintw(w[Win::Stats], 0, 0, "Route  : %s", qPrintable(replay->route()->name()));
  mvwprintw(w[Win::Stats], 1, 0, "Current: %d s  Total: %d s       ", cur_sec, total_sec);
  wrefresh(w[Win::Stats]);
}

void ConsoleUI::updateTimeline(int cur_sec, int total_sec) {
  auto draw_at = [=](int x, char c = ' ') {
    mvwaddch(w[Win::Timeline], 1, x, c);
    mvwaddch(w[Win::Timeline], 2, x, c);
  };
  werase(w[Win::Timeline]); 

  int width = getmaxx(w[Win::Timeline]);
  wattron(w[Win::Timeline], COLOR_PAIR(Color::Disengaged));
  for (int i = 0; i < width; ++i) {
    draw_at(i, ' ');
  }
  wattroff(w[Win::Timeline], COLOR_PAIR(Color::Disengaged));
  
  auto summary = replay->getSummary();
  for (auto [engage_sec, disengage_sec] : summary) {
    int start_pos = ((double)engage_sec/total_sec) * width;
    int end_pos = ((double)disengage_sec/total_sec) * width;
    wattron(w[Win::Timeline], COLOR_PAIR(Color::Engaged));
    for (int i = start_pos; i <= end_pos; ++i) {
      draw_at(i, ' ');
    }
    wattroff(w[Win::Timeline], COLOR_PAIR(Color::Engaged));
  }

  auto car_events = replay->getCarEvents();
  for (auto [start_sec, end_sec, status] : car_events) {
    int start_pos = ((double)start_sec / total_sec) * width;
    int end_pos = ((double)end_sec / total_sec) * width;
    wattron(w[Win::Timeline], COLOR_PAIR(Color::Alert));
    for (int i = start_pos; i <= end_pos; ++i) {
      draw_at(i, ' ');
    }
    wattroff(w[Win::Timeline], COLOR_PAIR(Color::Alert));
  }

  int cur_pos = ((double)cur_sec / total_sec) * width;
  wattron(w[Win::Timeline], COLOR_PAIR(Color::PosIndicator));
  mvwaddch(w[Win::Timeline], 0, cur_pos, ACS_VLINE);
  mvwaddch(w[Win::Timeline], 3, cur_pos, ACS_VLINE);
  wattroff(w[Win::Timeline], COLOR_PAIR(Color::PosIndicator));

  wrefresh(w[Win::Timeline]);
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
      replay->pause(true);
      getch_timer.stop();
      
      curs_set(1);
      nodelay(stdscr, false);
      int y = getmaxy(stdscr) - 8;
      mvprintw(y, 3, "Enter seek request: ");
      refresh();
      echo();
      int choice = 0;
      scanw((char *)"%d", &choice);
      noecho();
      nodelay(stdscr, true);
      curs_set(0);
      move(y, 0);
      clrtoeol(); 
      refresh();

      replay->pause(false);
      replay->seekTo(choice, false);
      getch_timer.start(1000, this);
      break;
    }
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
    case 'x':
      if (replay->hasFlag(REPLAY_FLAG_FULL_SPEED)) {
        replay->removeFlag(REPLAY_FLAG_FULL_SPEED);
        rInfo("replay at normal speed");
      } else {
        replay->addFlag(REPLAY_FLAG_FULL_SPEED);
        rInfo("replay at full speed");
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
