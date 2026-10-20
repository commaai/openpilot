#pragma once

#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

// the Logs tab: the messages of one id, newest first, as signal values or hex bytes
class LogsWidget {
public:
  LogsWidget();
  void setMessage(const MessageId &message_id);
  void updateState() { load(false); }  // appends what arrived since the last call
  void onShown() { load(true); }       // reloads the log when the Logs tab becomes visible
  void draw();

private:
  struct Message {
    uint64_t mono_time = 0;
    std::vector<double> sig_values;
    std::vector<uint8_t> data;
    std::vector<CabanaColor> colors;
  };

  bool hexMode() const { return sigs_.empty() || hex_mode_; }
  int columnCount() const { return hexMode() ? 2 : (int)sigs_.size() + 1; }
  void reset();
  void setFilter(int sig_idx, const std::string &value, std::function<bool(double, double)> cmp);
  void load(bool clear);
  bool canFetchMore() const;
  void fetch(std::deque<Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time);
  void filterChanged();
  void exportToCSV();
  void drawTable();
  std::string headerText(int column) const;
  ImVec2 headerSize(int column, float viewport_width) const;
  void drawHeaderCell(ImDrawList *dl, const ImRect &rect, int column) const;

  MessageId msg_id_;
  std::vector<cabana::Signal *> sigs_;
  std::deque<Message> messages_;
  CanData hex_colors_;
  bool hex_mode_ = false;
  int filter_sig_idx_ = -1;
  double filter_value_ = 0;
  std::function<bool(double, double)> filter_cmp_;
  int signals_cb_ = 0, comp_box_ = 0, display_type_cb_ = 0;  // current combo box indices
  std::string value_edit_;
  bool value_edit_modified_ = false;
  bool export_btn_enabled_ = false;
  int selected_row_ = -1, selected_col_ = -1;
  bool vscrollbar_visible_ = false;
  Connections connections_;
};
