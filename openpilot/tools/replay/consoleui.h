#pragma once

#include <array>
#include <mutex>
#include <vector>

#include "tools/replay/replay.h"
#include <ncurses.h>

class ConsoleUI {
public:
  ConsoleUI(Replay *replay);
  ~ConsoleUI();
  int exec();
  inline static const std::array speed_array = {0.2f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};

private:
  void initWindows();
  void handleKey(char c);
  void displayHelp();
  void displayTimelineDesc();
  void updateTimeline();
  void updateSummary();
  void updateStatus();
  void pauseReplay(bool pause);
  void updateSize();
  void updateProgressBar();
  void logMessage(ReplyMsgType type, const std::string &msg);

  enum Status { Playing, Paused };
  enum Win { Title, Stats, Log, LogBorder, DownloadBar, Timeline, TimelineDesc, Help, CarState, Max};
  std::array<WINDOW*, Win::Max> w{};
  SubMaster sm;
  Replay *replay;
  int max_width, max_height;
  Status status = Status::Playing;

  std::mutex mutex;
  std::vector<std::pair<ReplyMsgType, std::string>> logs;
  uint64_t progress_cur = 0;
  uint64_t progress_total = 0;
  bool download_success = false;
};
