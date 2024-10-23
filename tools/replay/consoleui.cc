#include "tools/replay/consoleui.h"

#include <initializer_list>
#include <string>
#include <tuple>
#include <utility>

#include <QApplication>

#include "common/ratekeeper.h"
#include "common/util.h"
#include "common/version.h"

namespace {

const int BORDER_SIZE = 3;

const std::initializer_list<std::pair<std::string, std::string>> keyboard_shortcuts[] = {
  {
    {"s", "+10s"},
    {"shift+s", "-10s"},
    {"m", "+60s"},
    {"shift+m", "-60s"},
    {"space", "Pause/Resume"},
    {"e", "Next Engagement"},
    {"d", "Next Disengagement"},
    {"t", "Next User Tag"},
    {"i", "Next Info"},
    {"w", "Next Warning"},
    {"c", "Next Critical"},
  },
  {
    {"enter", "Enter seek request"},
    {"+/-", "Playback speed"},
    {"q", "Exit"},
  },
};

enum Color {
  Default,
  Debug,
  Yellow,
  Green,
  Red,
  Cyan,
  BrightWhite,
  Engaged,
  Disengaged,
};

void add_str(WINDOW *w, const char *str, Color color = Color::Default, bool bold = false) {
  if (color != Color::Default) wattron(w, COLOR_PAIR(color));
  if (bold) wattron(w, A_BOLD);
  waddstr(w, str);
  if (bold) wattroff(w, A_BOLD);
  if (color != Color::Default) wattroff(w, COLOR_PAIR(color));
}

}  // namespace

ConsoleUI::ConsoleUI(Replay *replay) : replay(replay), sm({"carState", "liveParameters"}) {
  // Initialize curses
  initscr();
  clear();
  curs_set(false);
  cbreak();  // Line buffering disabled. pass on everything
  noecho();
  keypad(stdscr, true);
  nodelay(stdscr, true);  // non-blocking getchar()

  // Initialize all the colors. https://www.ditig.com/256-colors-cheat-sheet
  start_color();
  init_pair(Color::Debug, 246, COLOR_BLACK);  // #949494
  init_pair(Color::Yellow, 184, COLOR_BLACK);
  init_pair(Color::Red, COLOR_RED, COLOR_BLACK);
  init_pair(Color::Cyan, COLOR_CYAN, COLOR_BLACK);
  init_pair(Color::BrightWhite, 15, COLOR_BLACK);
  init_pair(Color::Disengaged, COLOR_BLUE, COLOR_BLUE);
  init_pair(Color::Engaged, 28, 28);
  init_pair(Color::Green, 34, COLOR_BLACK);

  initWindows();

  installMessageHandler([this](ReplyMsgType type, const std::string msg) {
    std::scoped_lock lock(mutex);
    logs.emplace_back(type, msg);
  });
  installDownloadProgressHandler([this](uint64_t cur, uint64_t total, bool success) {
    std::scoped_lock lock(mutex);
    progress_cur = cur;
    progress_total = total;
    download_success = success;
  });
}

ConsoleUI::~ConsoleUI() {
  endwin();
}

void ConsoleUI::initWindows() {
  getmaxyx(stdscr, max_height, max_width);
  w.fill(nullptr);
  w[Win::Title] = newwin(1, max_width, 0, 0);
  w[Win::Stats] = newwin(2, max_width - 2 * BORDER_SIZE, 2, BORDER_SIZE);
  w[Win::Timeline] = newwin(4, max_width - 2 * BORDER_SIZE, 5, BORDER_SIZE);
  w[Win::TimelineDesc] = newwin(1, 100, 10, BORDER_SIZE);
  w[Win::CarState] = newwin(3, 100, 12, BORDER_SIZE);
  w[Win::DownloadBar] = newwin(1, 100, 16, BORDER_SIZE);
  if (int log_height = max_height - 27; log_height > 4) {
    w[Win::LogBorder] = newwin(log_height, max_width - 2 * (BORDER_SIZE - 1), 17, BORDER_SIZE - 1);
    box(w[Win::LogBorder], 0, 0);
    w[Win::Log] = newwin(log_height - 2, max_width - 2 * BORDER_SIZE, 18, BORDER_SIZE);
    scrollok(w[Win::Log], true);
  }
  w[Win::Help] = newwin(5, max_width - (2 * BORDER_SIZE), max_height - 6, BORDER_SIZE);

  // set the title bar
  wbkgd(w[Win::Title], A_REVERSE);
  mvwprintw(w[Win::Title], 0, 3, "openpilot replay %s", COMMA_VERSION);

  // show windows on the real screen
  refresh();
  displayTimelineDesc();
  displayHelp();
  updateSummary();
  updateTimeline();
  for (auto win : w) {
    if (win) wrefresh(win);
  }
}

void ConsoleUI::updateSize() {
  if (is_term_resized(max_height, max_width)) {
    for (auto win : w) {
      if (win) delwin(win);
    }
    endwin();
    clear();
    refresh();
    initWindows();
    rWarning("resize term %dx%d", max_height, max_width);
  }
}

void ConsoleUI::updateStatus() {
  auto write_item = [this](int y, int x, const char *key, const std::string &value, const std::string &unit,
                           bool bold = false, Color color = Color::BrightWhite) {
    auto win = w[Win::CarState];
    wmove(win, y, x);
    add_str(win, key);
    add_str(win, value.c_str(), color, bold);
    add_str(win, unit.c_str());
  };
  static const std::pair<const char *, Color> status_text[] = {
      {"loading...", Color::Red},
      {"playing", Color::Green},
      {"paused...", Color::Yellow},
  };

  sm.update(0);

  auto [status_str, status_color] = status_text[status];
  write_item(0, 0, "STATUS:    ", status_str, "      ", false, status_color);
  std::string current_segment = " - " + std::to_string((int)(replay->currentSeconds() / 60));
  write_item(0, 25, "TIME:  ", replay->currentDateTime().toString("ddd MMMM dd hh:mm:ss").toStdString(), current_segment, true);

  auto p = sm["liveParameters"].getLiveParameters();
  write_item(1, 0, "STIFFNESS: ", util::string_format("%.2f %%", p.getStiffnessFactor() * 100), "  ");
  write_item(1, 25, "SPEED: ", util::string_format("%.2f", sm["carState"].getCarState().getVEgo()), " m/s");
  write_item(2, 0, "STEER RATIO: ", util::string_format("%.2f", p.getSteerRatio()), "");
  auto angle_offsets = util::string_format("%.2f|%.2f", p.getAngleOffsetAverageDeg(), p.getAngleOffsetDeg());
  write_item(2, 25, "ANGLE OFFSET(AVG|INSTANT): ", angle_offsets, " deg");

  wrefresh(w[Win::CarState]);
}

void ConsoleUI::displayHelp() {
  for (int i = 0; i < std::size(keyboard_shortcuts); ++i) {
    wmove(w[Win::Help], i * 2, 0);
    for (auto &[key, desc] : keyboard_shortcuts[i]) {
      wattron(w[Win::Help], A_REVERSE);
      waddstr(w[Win::Help], (' ' + key + ' ').c_str());
      wattroff(w[Win::Help], A_REVERSE);
      waddstr(w[Win::Help], (' ' + desc + ' ').c_str());
    }
  }
  wrefresh(w[Win::Help]);
}

void ConsoleUI::displayTimelineDesc() {
  std::tuple<Color, const char *, bool> indicators[]{
      {Color::Engaged, " Engaged ", false},
      {Color::Disengaged, " Disengaged ", false},
      {Color::Green, " Info ", true},
      {Color::Yellow, " Warning ", true},
      {Color::Red, " Critical ", true},
      {Color::Cyan, " User Tag ", true},
  };
  for (auto [color, name, bold] : indicators) {
    add_str(w[Win::TimelineDesc], "__", color, bold);
    add_str(w[Win::TimelineDesc], name);
  }
}

void ConsoleUI::logMessage(ReplyMsgType type, const std::string &msg) {
  if (auto win = w[Win::Log]) {
    Color color = Color::Default;
    if (type == ReplyMsgType::Debug) {
      color = Color::Debug;
    } else if (type == ReplyMsgType::Warning) {
      color = Color::Yellow;
    } else if (type == ReplyMsgType::Critical) {
      color = Color::Red;
    }
    add_str(win, (msg + "\n").c_str(), color);
    wrefresh(win);
  }
}

void ConsoleUI::updateProgressBar() {
  werase(w[Win::DownloadBar]);
  if (download_success && progress_cur < progress_total) {
    const int width = 35;
    const float progress = progress_cur / (double)progress_total;
    const int pos = width * progress;
    wprintw(w[Win::DownloadBar], "Downloading [%s>%s]  %d%% %s", std::string(pos, '=').c_str(),
            std::string(width - pos, ' ').c_str(), int(progress * 100.0), formattedDataSize(progress_total).c_str());
  }
  wrefresh(w[Win::DownloadBar]);
}

void ConsoleUI::updateSummary() {
  const auto &route = replay->route();
  mvwprintw(w[Win::Stats], 0, 0, "Route: %s, %lu segments", route->name().c_str(), route->segments().size());
  mvwprintw(w[Win::Stats], 1, 0, "Car Fingerprint: %s", replay->carFingerprint().c_str());
  wrefresh(w[Win::Stats]);
}

void ConsoleUI::updateTimeline() {
  auto win = w[Win::Timeline];
  int width = getmaxx(win);
  werase(win);

  wattron(win, COLOR_PAIR(Color::Disengaged));
  mvwhline(win, 1, 0, ' ', width);
  mvwhline(win, 2, 0, ' ', width);
  wattroff(win, COLOR_PAIR(Color::Disengaged));

  const int total_sec = replay->maxSeconds() - replay->minSeconds();
  for (auto [begin, end, type] : replay->getTimeline()) {
    int start_pos = ((begin - replay->minSeconds()) / total_sec) * width;
    int end_pos = ((end - replay->minSeconds()) / total_sec) * width;
    if (type == TimelineType::Engaged) {
      mvwchgat(win, 1, start_pos, end_pos - start_pos + 1, A_COLOR, Color::Engaged, NULL);
      mvwchgat(win, 2, start_pos, end_pos - start_pos + 1, A_COLOR, Color::Engaged, NULL);
    } else if (type == TimelineType::UserFlag) {
      mvwchgat(win, 3, start_pos, end_pos - start_pos + 1, ACS_S3, Color::Cyan, NULL);
    } else {
      auto color_id = Color::Green;
      if (type != TimelineType::AlertInfo) {
        color_id = type == TimelineType::AlertWarning ? Color::Yellow : Color::Red;
      }
      mvwchgat(win, 3, start_pos, end_pos - start_pos + 1, ACS_S3, color_id, NULL);
    }
  }

  int cur_pos = ((replay->currentSeconds() - replay->minSeconds()) / total_sec) * width;
  wattron(win, COLOR_PAIR(Color::BrightWhite));
  mvwaddch(win, 0, cur_pos, ACS_VLINE);
  mvwaddch(win, 3, cur_pos, ACS_VLINE);
  wattroff(win, COLOR_PAIR(Color::BrightWhite));
  wrefresh(win);
}

void ConsoleUI::pauseReplay(bool pause) {
  replay->pause(pause);
  status = pause ? Status::Paused : Status::Playing;
}

void ConsoleUI::handleKey(char c) {
  if (c == '\n') {
    // pause the replay and blocking getchar()
    pauseReplay(true);
    updateStatus();
    curs_set(true);
    nodelay(stdscr, false);

    // Wait for user input
    rWarning("Waiting for input...");
    int y = getmaxy(stdscr) - 9;
    move(y, BORDER_SIZE);
    add_str(stdscr, "Enter seek request: ", Color::BrightWhite, true);
    refresh();

    // Seek to choice
    echo();
    int choice = 0;
    scanw((char *)"%d", &choice);
    noecho();
    pauseReplay(false);
    replay->seekTo(choice, false);

    // Clean up and turn off the blocking mode
    move(y, 0);
    clrtoeol();
    nodelay(stdscr, true);
    curs_set(false);
    refresh();

  } else if (c == '+' || c == '=') {
    auto it = std::upper_bound(speed_array.begin(), speed_array.end(), replay->getSpeed());
    if (it != speed_array.end()) {
      rWarning("playback speed: %.1fx", *it);
      replay->setSpeed(*it);
    }
  } else if (c == '_' || c == '-') {
    auto it = std::lower_bound(speed_array.begin(), speed_array.end(), replay->getSpeed());
    if (it != speed_array.begin()) {
      auto prev = std::prev(it);
      rWarning("playback speed: %.1fx", *prev);
      replay->setSpeed(*prev);
    }
  } else if (c == 'e') {
    replay->seekToFlag(FindFlag::nextEngagement);
  } else if (c == 'd') {
    replay->seekToFlag(FindFlag::nextDisEngagement);
  } else if (c == 't') {
    replay->seekToFlag(FindFlag::nextUserFlag);
  } else if (c == 'i') {
    replay->seekToFlag(FindFlag::nextInfo);
  } else if (c == 'w') {
    replay->seekToFlag(FindFlag::nextWarning);
  } else if (c == 'c') {
    replay->seekToFlag(FindFlag::nextCritical);
  } else if (c == 'm') {
    replay->seekTo(+60, true);
  } else if (c == 'M') {
    replay->seekTo(-60, true);
  } else if (c == 's') {
    replay->seekTo(+10, true);
  } else if (c == 'S') {
    replay->seekTo(-10, true);
  } else if (c == ' ') {
    pauseReplay(!replay->isPaused());
  }
}

int ConsoleUI::exec() {
  RateKeeper rk("Replay", 20);
  while (true) {
    int c = getch();
    if (c == 'q' || c == 'Q') {
      break;
    }
    handleKey(c);

    if (rk.frame() % 25) {
      updateSize();
      updateSummary();
    }

    updateTimeline();
    updateStatus();

    {
      std::scoped_lock lock(mutex);
      updateProgressBar();
      for (auto &[type, msg] : logs) {
        logMessage(type, msg);
      }
      logs.clear();
    }

    qApp->processEvents();
    rk.keepTime();
  }
  return 0;
}
