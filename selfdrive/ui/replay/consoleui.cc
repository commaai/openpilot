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
  installMessageHandler(std::bind(&ConsoleUI::LogMessage, this, _1, _2));
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
  init_pair(Color::bgTimeLine, COLOR_BLACK, 8);
  init_pair(Color::bgTitle, COLOR_BLACK, COLOR_WHITE);
  init_pair(Color::Played, COLOR_GREEN, COLOR_GREEN);
  init_pair(Color::Engaged, COLOR_BLUE, COLOR_BLUE);
  init_pair(Color::EngagedPlayed, 22, 22);
  init_pair(Color::CarEvent, 10, 10);
  init_pair(Color::CarEventPlayed, COLOR_YELLOW, COLOR_YELLOW);
  init_pair(Color::AlertWarning, COLOR_RED, COLOR_RED);

  int height, width;
  getmaxyx(stdscr, height, width);

  w[Win::Title] = newwin(1, width , 0, 0);
  w[Win::Stats] = newwin(2, width, 2, 3);
  w[Win::Timeline] = newwin(2, 100, 5, 3);
  w[Win::TimelineDesc]= newwin(1, 100, 8, 3);
  w[Win::DownloadBar] = newwin(1, 100, 10, 3);
  w[Win::CarState] = newwin(5, 100, 11, 3);
  w[Win::LogBorder] = newwin(height - 30, 100, 17, 2);
  w[Win::Log] = newwin(height - 34, 98, 18, 3);
  w[Win::Help] = newwin(9, 100, height-10, 3);
  
  wbkgd(w[Win::Title], COLOR_PAIR(Color::bgTitle));
  scrollok(w[Win::Log], true);
  box(w[Win::LogBorder], 0, 0);
  wbkgd(w[Win::Timeline], COLOR_PAIR(Color::bgTimeLine));
  
  refresh();

  mvwprintw(w[Win::Title], 0, 3, "openpilot replay %s", COMMA_VERSION);

  std::pair<Color, const char *> indicators[] {
    {Color::Engaged, " engaged "},
    {Color::bgTimeLine, " disengaged "},
    {Color::CarEvent, " alert "},
    {Color::AlertWarning, " warning "},
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
  wrefresh(w[Win::LogBorder]);
  displayHelp();
  updateStats(0, replay->route()->segments().size() * 60);

  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateTimeline);
  QObject::connect(replay, &Replay::updateProgress, this, &ConsoleUI::updateStats);
  QObject::connect(&m_notifier, SIGNAL(activated(int)), SLOT(readyRead()));
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

void ConsoleUI::LogMessage(ReplyMsgType type, const char *msg) {
  if (w[Win::Log]) {
    wattron(w[Win::Log], COLOR_PAIR((int)type));
    wprintw(w[Win::Log], "%s\n", msg);
    wattroff(w[Win::Log], COLOR_PAIR((int)type));
    wrefresh(w[Win::Log]);
  }
}

void ConsoleUI::displayHelp() {
  std::initializer_list<std::pair<const char *, const char*>> single_line_keys {
    {"s", "+10s"},
    {"shift+s", "-10s"},
    {"m", "+60s"},
    {"shift+m", "+60s"},
    {"p", "pause/resume"},
  };
  std::initializer_list<std::pair<const char *, const char*>> multi_line_keys = {
    {"e", "next engagement"},
    {"d", "next disengagement"},
    {"enter", "enter seek request"},
    {"x", "replay at full speed"},
    {"q", "exit"},
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

void ConsoleUI::downloadProgressHandler(uint64_t cur, uint64_t total) {
  if (w[Win::DownloadBar]) {
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
}

void ConsoleUI::updateStats(int cur_sec, int total_sec) {
  mvwprintw(w[Win::Stats], 0, 0, "Route  : %s", qPrintable(replay->route()->name()));
  mvwprintw(w[Win::Stats], 1, 0, "Current: %d s  Total: %d s       ", cur_sec, total_sec);
  wrefresh(w[Win::Stats]);
}

void ConsoleUI::updateTimeline(int cur_sec, int total_sec) {
  auto draw_at = [=](int x, char c = ' ') {
    mvwaddch(w[Win::Timeline], 0, x, c);
    mvwaddch(w[Win::Timeline], 1, x, c);
  };
  werase(w[Win::Timeline]); 

  int width = getmaxx(w[Win::Timeline]);
  int cur_pos = ((double)cur_sec / total_sec) * width;
  wattron(w[Win::Timeline], COLOR_PAIR(Color::Played));
  for (int i = 0; i <= cur_pos; ++i) {
    draw_at(i, ' ');
  }
  wattroff(w[Win::Timeline], COLOR_PAIR(Color::Played));
  
  auto summary = replay->getSummary();
  for (auto [engage_sec, disengage_sec] : summary) {
    int start_pos = ((double)engage_sec/total_sec) * width;
    int end_pos = ((double)disengage_sec/total_sec) * width;
    for (int i = start_pos; i <= end_pos; ++i) {
      wattron(w[Win::Timeline], COLOR_PAIR((int)(i < cur_pos ? Color::EngagedPlayed : Color::Engaged)));
      draw_at(i, ' ');
      wattroff(w[Win::Timeline], COLOR_PAIR((int)(i < cur_pos ? Color::EngagedPlayed : Color::Engaged)));
    }
  }

  auto car_events = replay->getCarEvents();
  for (auto [start_sec, end_sec, status] : car_events) {
    int start_pos = ((double)start_sec / total_sec) * width;
    int end_pos = ((double)end_sec / total_sec) * width;
    for (int i = start_pos; i <= end_pos; ++i) {
      wattron(w[Win::Timeline], COLOR_PAIR(int(i < cur_pos ? Color::CarEventPlayed : Color::CarEvent)));
      draw_at(i, ' ');
      wattroff(w[Win::Timeline], COLOR_PAIR(int(i < cur_pos ? Color::CarEventPlayed : Color::CarEvent)));
    }
  }

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
      int y = getmaxy(stdscr) - 12;
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
