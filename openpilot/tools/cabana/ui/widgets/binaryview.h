#pragma once

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/core/observable.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class BinaryView;
class BinaryViewModel;

// QModelIndex of the Qt table view: a (row, column) into BinaryViewModel::items
struct BinaryIndex {
  int row = -1;
  int column = -1;
  bool isValid() const { return row >= 0 && column >= 0; }
  bool operator==(const BinaryIndex &o) const { return row == o.row && column == o.column; }
  bool operator!=(const BinaryIndex &o) const { return !(*this == o); }
  bool operator<(const BinaryIndex &o) const { return std::tie(row, column) < std::tie(o.row, o.column); }
};

class BinaryItemDelegate {
public:
  BinaryItemDelegate(BinaryView *parent);
  void paint(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index) const;
  bool hasSignal(const BinaryIndex &index, int dx, int dy, const cabana::Signal *sig) const;
  void drawSignalCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index, const cabana::Signal *sig) const;

  const float small_font_size = 8.0f;  // small_font.setPixelSize(8)
  std::array<std::string, 256> hex_text_table;
  std::array<std::string, 2> bin_text_table;

private:
  BinaryView *bin_view;
};

class BinaryViewModel {
public:
  BinaryViewModel() = default;
  void refresh();
  void updateState();
  void updateItem(int row, int col, uint8_t val, const CabanaColor &color);
  std::string headerData(int section) const;      // vertical header, Qt::DisplayRole
  std::string data(const BinaryIndex &index) const;  // Qt::ToolTipRole
  int rowCount() const { return row_count; }
  int columnCount() const { return column_count; }
  BinaryIndex index(int row, int column) const { return {row, column}; }
  // Qt::ItemIsSelectable
  bool isSelectable(const BinaryIndex &index) const { return index.column != column_count - 1; }
  const std::vector<std::array<uint32_t, 8>> &getBitFlipChanges(size_t msg_size);

  struct BitFlipTracker {
    std::optional<std::pair<double, double>> time_range;
    std::vector<std::array<uint32_t, 8>> flip_counts;
  } bit_flip_tracker;

  struct Item {
    CabanaColor bg_color = CabanaColor(102, 86, 169, 255);
    bool is_msb = false;
    bool is_lsb = false;
    uint8_t val;
    std::vector<const cabana::Signal *> sigs;
    bool valid = false;
  };
  std::vector<Item> items;
  bool heatmap_live_mode = true;
  MessageId msg_id;
  int row_count = 0;
  const int column_count = 9;
};

class BinaryView {
public:
  BinaryView();
  void setMessage(const MessageId &message_id);
  void highlight(const cabana::Signal *sig);
  std::set<const cabana::Signal*> getOverlappingSignals() const;
  void updateState() { model->updateState(); }
  // paintEvent + the mouse/keyboard events; draws inline into the current (scrollable) window
  void draw();
  ImVec2 minimumSizeHint() const;
  void setHeatmapLiveMode(bool live) { model->heatmap_live_mode = live; updateState(); }
  std::string whatsThis() const;

  Observable<const cabana::Signal *> signalClicked;
  Observable<const cabana::Signal *> signalHovered;
  Observable<const cabana::Signal *, cabana::Signal &> editSignal;
  Observable<const MessageId &, const cabana::Signal *, bool, bool> showChart;

private:
  void addShortcuts();  // polled every frame from draw()
  void refresh();
  std::tuple<int, int, bool> getSelection(BinaryIndex index);
  void setSelection();
  void mousePressEvent(const ImVec2 &pos);
  void mouseMoveEvent(const ImVec2 &pos);
  void mouseReleaseEvent(const ImVec2 &pos);
  void leaveEvent();
  void highlightPosition(const ImVec2 &pt);
  // QTableView geometry/selection
  BinaryIndex indexAt(const ImVec2 &pos) const;
  ImRect visualRect(const BinaryIndex &index) const;
  bool hasSelection() const { return !selection_.empty(); }
  void clearSelection() { selection_.clear(); }
  bool isSelected(const BinaryIndex &index) const { return selection_.count(index) > 0; }

  BinaryIndex anchor_index;
  ImVec2 last_mouse_pos{-1, -1};
  std::unique_ptr<BinaryViewModel> model;
  std::unique_ptr<BinaryItemDelegate> delegate;
  bool is_message_active = false;
  const cabana::Signal *resize_sig = nullptr;
  const cabana::Signal *hovered_sig = nullptr;
  Connections connections_;
  std::set<BinaryIndex> selection_;  // selectionModel()
  ImVec2 grid_pos_;                  // viewport origin of the current frame
  float column_width_ = 0;           // horizontalHeader() Stretch section size
  bool under_mouse_ = false;         // underMouse()
  bool scroll_to_top_ = false;       // verticalScrollBar()->setValue(0)
  friend class BinaryItemDelegate;
};
