#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/tools/tooldialog.h"

class FindSignalModel {
public:
  struct SearchSignal {
    MessageId id = {};
    uint64_t mono_time = 0;
    cabana::Signal sig = {};
    double value = 0.;
    std::vector<std::string> values;
  };

  FindSignalModel() {}
  std::string headerData(int section, bool horizontal) const;
  std::string data(int row, int column) const;
  int columnCount() const { return 3; }
  int rowCount() const { return std::min((int)filtered_signals.size(), 300); }
  void search(std::function<bool(double)> cmp);
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
  bool draw() override;

  Observable<const MessageId &> openMessage;

private:
  void search();
  void modelReset();
  void setInitialSignals();
  void customMenuRequested(int row);
  void drawMessageGroup();
  void drawPropertiesGroup();
  void drawFindGroup();
  void drawTable();

  std::string value1, value2, factor_edit = "1.0", offset_edit = "0.0";
  std::string bus_edit, address_edit, first_time_edit = "0", last_time_edit = "MAX";
  int compare_cb = 0;
  int min_size = 8, max_size = 8;
  bool litter_endian = true, is_signed = false;
  bool search_btn_enabled = true, reset_btn_enabled = false, undo_btn_enabled = false;
  std::string search_btn_text = "Find";
  bool properties_group_enabled = true, message_group_enabled = true;
  bool to_label_visible = false, stats_label_visible = false;
  std::string stats_label;
  std::unique_ptr<FindSignalModel> model;
  std::string title_;
  bool open_ = true;
  std::function<bool(double)> pending_cmp_;  // deferred to the next frame so "Finding ...." paints first
};
