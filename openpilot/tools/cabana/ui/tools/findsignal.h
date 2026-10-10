#pragma once

#include <algorithm>
#include <functional>
#include <future>
#include <limits>
#include <string>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/tools/tooldialog.h"

struct SignalSearch {
  struct SearchSignal {
    MessageId id = {};
    uint64_t mono_time = 0;
    cabana::Signal sig = {};
    double value = 0.;
    std::vector<std::string> values;
  };

  void search(const std::function<bool(double)> &cmp);
  void reset();
  void undo();

  std::vector<SearchSignal> filtered_signals;
  std::vector<SearchSignal> initial_signals;
  std::vector<std::vector<SearchSignal>> histories;
  uint64_t last_time = std::numeric_limits<uint64_t>::max();
};

class FindSignalDlg : public ToolDialog {
public:
  FindSignalDlg();
  ~FindSignalDlg() override;
  bool draw() override;

  Observable<const MessageId &> openMessage;

private:
  void search();
  void setInitialSignals();
  void drawContextMenu(int row);
  void drawMessageGroup();
  void drawPropertiesGroup();
  void drawFindGroup();
  void drawTable();

  std::string value1_, value2_, factor_ = "1.0", offset_ = "0.0";
  std::string bus_, address_, first_time_ = "0", last_time_ = "MAX";
  int compare_ = 0;
  int min_size_ = 8, max_size_ = 8;
  bool little_endian_ = true, is_signed_ = false;
  bool searched_ = false;  // a search/undo/reset ran, so the stats line is shown
  SignalSearch search_;
  std::future<void> search_future_;  // search_ is off limits while it is valid
  bool searching_ = false;
};
