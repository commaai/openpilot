#pragma once

#include <array>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/icons.h"
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

  MessageId msg_id;
  CanData hex_colors;
  const int batch_size = 50;
  int filter_sig_idx = -1;
  double filter_value = 0;
  std::function<bool(double, double)> filter_cmp = nullptr;
  std::deque<Message> messages;
  std::vector<cabana::Signal *> sigs;
  bool hex_mode = false;
  Connections connections_;
};

class LogsWidget {
public:
  LogsWidget();
  void setMessage(const MessageId &message_id) {
    selected_row = -1;
    model.setMessage(message_id);
  }
  void updateState() { model.updateState(); }
  void showEvent() { model.updateState(true); }  // call when the Logs tab becomes visible
  void draw();

private:
  void filterChanged();
  void exportToCSV();
  void modelReset();
  void drawTable();

  HistoryLogModel model;
  int signals_cb = 0, comp_box = 0, display_type_cb = 0;  // current combo box indices
  std::string value_edit;
  bool value_edit_modified = false;
  bool filters_widget_visible = true;
  bool export_btn_enabled = false;
  int selected_row = -1, selected_col = -1;
  bool vscrollbar_visible_ = false;
  MessageBytesDelegate delegate;
  Connections connections_;
};
