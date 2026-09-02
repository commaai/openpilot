#pragma once

#include <array>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "imgui.h"

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/widgets/messagebytes.h"

class HistoryLogModel {
public:
  HistoryLogModel();
  void setMessage(const MessageId &message_id);
  void updateState(bool clear = false);
  void setFilter(int sig_idx, const std::string &value, std::function<bool(double, double)> cmp);
  std::string headerData(int section) const;
  std::optional<CabanaColor> headerBackground(int section) const;
  std::string data(int row, int col) const;  // the byte colors and bytes are read from messages[row] directly
  void fetchMore();
  bool canFetchMore() const;
  int rowCount() const { return messages.size(); }
  int columnCount() const { return !isHexMode() ? sigs.size() + 1 : 2; }
  inline bool isHexMode() const { return sigs.empty() || hex_mode; }
  void reset();
  void setHexMode(bool hex_mode);

  struct Message {
    uint64_t mono_time = 0;
    std::vector<double> sig_values;
    std::vector<uint8_t> data;
    std::vector<CabanaColor> colors;
  };

  void fetchData(std::deque<Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time);

  Observable<> modelReset;
  Observable<int, int> rowsInserted;  // position, count
  Observable<> rowsRemoved;

  MessageId msg_id;
  CanData hex_colors;
  const int batch_size = 50;
  int filter_sig_idx = -1;
  double filter_value = 0;
  std::function<bool(double, double)> filter_cmp = nullptr;
  std::deque<Message> messages;
  std::vector<cabana::Signal *> sigs;
  bool hex_mode = false;

private:
  Connections connections_;
};

class LogsWidget {
public:
  LogsWidget();
  void setMessage(const MessageId &message_id) {
    selected_row_ = -1;
    model_.setMessage(message_id);
  }
  void updateState() { model_.updateState(); }
  void onShown() { model_.updateState(true); }  // reloads the log when the Logs tab becomes visible
  void draw();

private:
  void filterChanged();
  void exportToCSV();
  void modelReset();
  void drawTable();

  HistoryLogModel model_;
  int signals_cb_ = 0, comp_box_ = 0, display_type_cb_ = 0;  // current combo box indices
  std::string value_edit_;
  bool value_edit_modified_ = false;
  bool filters_widget_visible_ = true;
  bool export_btn_enabled_ = false;
  int selected_row_ = -1, selected_col_ = -1;
  bool vscrollbar_visible_ = false;
  MessageBytesDelegate delegate_;
  Connections connections_;
};
