#include "selfdrive/ui/replay/consoleui.h"

#include <QApplication>
#include <initializer_list>
#include <iostream>

#include "selfdrive/common/params.h"
#include "selfdrive/common/version.h"

enum Color {
  Info,
  Debug,
  Warning,
  Critical,
  bgWhite,
  BrightWhite,
  Engaged,
  Disengaged,
};

ConsoleUI::ConsoleUI(Replay *replay, QObject *parent) : replay(replay), sm({"carState", "liveParameters"}), QObject(parent) {
  qRegisterMetaType<uint64_t>("uint64_t");
  installMessageHandler([this](ReplyMsgType type, const std::string msg) {
    emit logMessageSignal((int)type, QString::fromStdString(msg));
  });
  installDownloadProgressHandler([this](uint64_t cur, uint64_t total, bool success) {
    emit updateProgressBarSignal(cur, total, success);
  });

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
  init_pair(Color::bgWhite, COLOR_BLACK, 15);
  init_pair(Color::Engaged, 28, 28);
  init_pair(Color::BrightWhite, 15, COLOR_BLACK);

  int height, width;
  getmaxyx(stdscr, height, width);

  w[Win::Title] = newwin(1, width, 0, 0);
  w[Win::Stats] = newwin(2, width, 2, 3);
  w[Win::Timeline] = newwin(4, 100, 5, 3);
  w[Win::TimelineDesc] = newwin(1, 100, 10, 3);
  w[Win::CarState] = newwin(3, 100, 12, 3);
  w[Win::DownloadBar] = newwin(1, 100, 16, 3);
  if (int log_height = height - 27; log_height > 5) {
    w[Win::LogBorder] = newwin(log_height, 100, 17, 2);
    box(w[Win::LogBorder], 0, 0);
    w[Win::Log] = newwin(log_height - 2, 98, 18, 3);
    scrollok(w[Win::Log], true);
  }
  w[Win::Help] = newwin(5, 100, height - 6, 3);

  wbkgd(w[Win::Title], COLOR_PAIR(Color::bgWhite));
  mvwprintw(w[Win::Title], 0, 3, "openpilot replay %s", COMMA_VERSION);

  std::tuple<Color, const char *, bool> indicators[]{
      {Color::Engaged, " Engaged ", false},
      {Color::Disengaged, " Disengaged ", false},
      {Color::Warning, " Alert ", true},
      {Color::Critical, " Warning ", true},
  };
  for (auto [color, name, bold] : indicators) {
    wattron(w[Win::TimelineDesc], COLOR_PAIR(color));
    if (bold) wattron(w[Win::TimelineDesc], A_BOLD);
    waddstr(w[Win::TimelineDesc], "__");
    if (bold) wattroff(w[Win::TimelineDesc], A_BOLD);
    wattroff(w[Win::TimelineDesc], COLOR_PAIR(color));
    waddstr(w[Win::TimelineDesc], name);
  }

  refresh();
  displayHelp();
  updateSummary();
  updateTimeline(0, replay->route()->segments().size() * 60);

  for (auto win : w) {
    wrefresh(win);
  }

  QObject::connect(replay, &Replay::updateTime, this, &ConsoleUI::updateTimeline);
  QObject::connect(replay, &Replay::streamStarted, this, &ConsoleUI::updateSummary);
  QObject::connect(&notifier, SIGNAL(activated(int)), SLOT(readyRead()));
  QObject::connect(this, &ConsoleUI::updateProgressBarSignal, this, &ConsoleUI::updateProgressBar);
  QObject::connect(this, &ConsoleUI::logMessageSignal, this, &ConsoleUI::logMessage);

  sm_timer.callOnTimeout(this, &ConsoleUI::updateStatus);
  sm_timer.start(50);
  getch_timer.start(1000, this);
  readyRead();
}

ConsoleUI::~ConsoleUI() {
  endwin();
}

void ConsoleUI::timerEvent(QTimerEvent *ev) {
  if (ev->timerId() != getch_timer.timerId()) return;
  refresh();
}

void ConsoleUI::updateStatus() {
  auto write_item = [this](int y, int x, const char *key, const std::string &value, const char *unit) {
    auto win = w[Win::CarState];
    mvwaddstr(win, y, x, key);
    wattron(win, COLOR_PAIR(Color::BrightWhite));
    wattron(win, A_BOLD);
    waddstr(win, value.c_str());
    wattroff(win, A_BOLD);
    wattroff(win, COLOR_PAIR(Color::BrightWhite));
    waddstr(win, unit);
  };

  write_item(0, 0, "SECONDS:   ", std::to_string(replay->currentSeconds()), " s     ");
  sm.update(0);
  if (sm.updated("carState")) {
    write_item(0, 25, "SPEED:   ", util::string_format("%.2f", sm["carState"].getCarState().getVEgo()), " m/s");
  }
  if (sm.updated("liveParameters")) {
    auto p = sm["liveParameters"].getLiveParameters();
    write_item(1, 0, "STIFFNESS: ", util::string_format("%.2f", p.getStiffnessFactor() * 100), " deg");
    write_item(1, 25, "STEER RATIO: ", util::string_format("%.2f", p.getSteerRatio()), "");
    auto angle_offsets = util::string_format("%.2f|%.2f", p.getAngleOffsetAverageDeg(), p.getAngleOffsetDeg());
    write_item(2, 0, "ANGLE OFFSET(AVG|INSTANCE): ", angle_offsets, " deg");
  }
  wrefresh(w[Win::CarState]);
}

void ConsoleUI::displayHelp() {
  std::initializer_list<std::pair<const char *, const char *>> single_line_keys{
      {"s", "+10s"},
      {"shift+s", "-10s"},
      {"m", "+60s"},
      {"shift+m", "+60s"},
      {"p", "Pause/Resume"},
      {"e", "Next Engmt"},
      {"d", "Next DisEngmt"},
  };
  std::initializer_list<std::pair<const char *, const char *>> multi_line_keys = {
      {"enter", "Enter seek request"},
      {"x", "Replay at full speed"},
      {"q", "Exit"},
  };

  auto write_shortcut = [=](std::string key, std::string desc) {
    wattron(w[Win::Help], COLOR_PAIR(Color::bgWhite));
    waddstr(w[Win::Help], (' ' + key + ' ').c_str());
    wattroff(w[Win::Help], COLOR_PAIR(Color::bgWhite));
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

void ConsoleUI::logMessage(int type, const QString &msg) {
  if (auto win = w[Win::Log]) {
    wattron(win, COLOR_PAIR((int)type));
    wprintw(win, "%s\n", qPrintable(msg));
    wattroff(win, COLOR_PAIR((int)type));
    wrefresh(win);
  }
}

void ConsoleUI::updateProgressBar(uint64_t cur, uint64_t total, bool success) {
  werase(w[Win::DownloadBar]);
  if (success && cur < total) {
    const int width = 30;
    const float progress = cur / (double)total;
    const int pos = width * progress;
    std::string s = util::string_format("Downloading [%s>%s]  %d%% %s", std::string(pos, '=').c_str(),
                                        std::string(width - pos, ' ').c_str(), int(progress * 100.0),
                                        formattedDataSize(total).c_str());
    waddstr(w[Win::DownloadBar], s.c_str());
  }
  wrefresh(w[Win::DownloadBar]);
}

void ConsoleUI::updateSummary() {
  const auto &route = replay->route();
  mvwprintw(w[Win::Stats], 0, 0, "Route: %s, %d segments", qPrintable(route->name()), route->segments().size());
  mvwprintw(w[Win::Stats], 1, 0, "Car Name: %s", replay->carName().c_str());
  wrefresh(w[Win::Stats]);
}

void ConsoleUI::updateTimeline(int cur_sec, int total_sec) {
  auto win = w[Win::Timeline];
  int width = getmaxx(win);
  werase(win);

  wattron(win, COLOR_PAIR(Color::Disengaged));
  for (int i = 0; i < width; ++i) {
    mvwaddch(win, 1, i, ' ');
    mvwaddch(win, 2, i, ' ');
  }
  wattroff(win, COLOR_PAIR(Color::Disengaged));

  auto summary = replay->getTimeline();
  for (auto [engage_sec, disengage_sec] : summary) {
    int start_pos = ((double)engage_sec / total_sec) * width;
    int end_pos = ((double)disengage_sec / total_sec) * width;
    wattron(win, COLOR_PAIR(Color::Engaged));
    for (int i = start_pos; i <= end_pos; ++i) {
      mvwaddch(win, 1, i, ' ');
      mvwaddch(win, 2, i, ' ');
    }
    wattroff(win, COLOR_PAIR(Color::Engaged));
  }

  auto car_events = replay->getCarEvents();
  for (auto [start_sec, end_sec, status] : car_events) {
    int start_pos = ((double)start_sec / total_sec) * width;
    int end_pos = ((double)end_sec / total_sec) * width;
    const bool critical = status == cereal::ControlsState::AlertStatus::CRITICAL;
    wattron(win, COLOR_PAIR(critical ? Color::Critical : Color::Warning));
    wattron(win, A_BOLD);
    for (int i = start_pos; i <= end_pos; ++i) {
      mvwaddch(win, 3, i, ACS_S3);
    }
    wattroff(win, A_BOLD);
    wattroff(win, COLOR_PAIR(critical ? Color::Critical : Color::Warning));
  }

  int cur_pos = ((double)cur_sec / total_sec) * width;
  wattron(win, COLOR_PAIR(Color::BrightWhite));
  mvwaddch(win, 0, cur_pos, ACS_VLINE);
  mvwaddch(win, 3, cur_pos, ACS_VLINE);
  wattroff(win, COLOR_PAIR(Color::BrightWhite));

  wrefresh(win);
}

void ConsoleUI::readyRead() {
  int c;
  while ((c = getch()) != ERR) {
    handleKey(c);
  }
}

void ConsoleUI::handleKey(char c) {
  switch (c) {
    case '\n': {
      replay->pause(true);
      getch_timer.stop();
      curs_set(1);
      nodelay(stdscr, false);

      rWarning("Waiting for input...");
      int y = getmaxy(stdscr) - 9;
      attron(COLOR_PAIR(Color::BrightWhite));
      attron(A_BOLD);
      mvprintw(y, 3, "Enter seek request: ");
      attroff(A_BOLD);
      attroff(COLOR_PAIR(Color::BrightWhite));
      // mvchgat(y, 1, -1, A_BLINK, 1, NULL);
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
        rWarning("replay at normal speed");
      } else {
        replay->addFlag(REPLAY_FLAG_FULL_SPEED);
        rWarning("replay at full speed");
      }
      break;
    case ' ':
      replay->pause(!replay->isPaused());
      break;
    case 'q':
    case 'Q':
      replay->stop();
      qApp->exit();
      break;
  }
}
