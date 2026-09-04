#pragma once

#include <array>
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

// a (row, column) of the bit grid: 8 bit columns and the hex column
struct BinaryIndex {
  int row = -1;
  int column = -1;
  bool isValid() const { return row >= 0 && column >= 0; }
  bool operator==(const BinaryIndex &o) const { return row == o.row && column == o.column; }
  bool operator<(const BinaryIndex &o) const { return std::tie(row, column) < std::tie(o.row, o.column); }
};

class BinaryView {
public:
  static constexpr int COLUMN_COUNT = 9;
  static constexpr int HEX_COLUMN = 8;

  BinaryView();
  void setMessage(const MessageId &message_id);
  void highlight(const cabana::Signal *sig);
  std::set<const cabana::Signal *> getOverlappingSignals() const;
  void updateState();
  // draws inline into the current (scrollable) window and handles the mouse/keyboard
  void draw();
  ImVec2 minimumSizeHint() const;
  void setHeatmapLiveMode(bool live) { heatmap_live_mode_ = live; updateState(); }
  std::string whatsThis() const;

  Observable<const cabana::Signal *> signalClicked;
  Observable<const cabana::Signal *> signalHovered;
  Observable<const cabana::Signal *, cabana::Signal &> editSignal;
  Observable<const MessageId &, const cabana::Signal *, bool, bool> showChart;

private:
  struct Cell {
    CabanaColor bg_color = CabanaColor(102, 86, 169, 255);
    bool is_msb = false;
    bool is_lsb = false;
    uint8_t val;
    std::vector<const cabana::Signal *> sigs;
    bool valid = false;
  };

  void refresh();  // rebuilds the grid from the DBC message
  void setCell(int row, int col, uint8_t val, const CabanaColor &color);
  const std::vector<std::array<uint32_t, 8>> &bitFlipChanges(size_t msg_size);
  Cell &cellAt(const BinaryIndex &index) { return cells_[index.row * COLUMN_COUNT + index.column]; }
  const Cell &cellAt(const BinaryIndex &index) const { return cells_[index.row * COLUMN_COUNT + index.column]; }

  void addShortcuts();  // polled every frame from draw()
  std::tuple<int, int, bool> getSelection(BinaryIndex index);
  void setSelection();
  void handleMousePress(const ImVec2 &pos);
  void handleMouseMove(const ImVec2 &pos);
  void handleMouseRelease(const ImVec2 &pos);
  void highlightPosition(const ImVec2 &pt);
  BinaryIndex indexAt(const ImVec2 &pos) const;
  ImRect visualRect(const BinaryIndex &index) const;
  bool hasSelection() const { return !selection_.empty(); }
  bool isSelected(const BinaryIndex &index) const { return selection_.count(index) > 0; }

  void paintCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index) const;
  bool hasSignal(const BinaryIndex &index, int dx, int dy, const cabana::Signal *sig) const;
  void drawSignalCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index, const cabana::Signal *sig) const;

  MessageId msg_id_;
  std::vector<Cell> cells_;
  int row_count_ = 0;
  bool heatmap_live_mode_ = true;
  struct BitFlipTracker {
    std::optional<std::pair<double, double>> time_range;
    std::vector<std::array<uint32_t, 8>> flip_counts;
  } bit_flip_tracker_;

  BinaryIndex anchor_index_;
  ImVec2 last_mouse_pos_{-1, -1};
  bool is_message_active_ = false;
  const cabana::Signal *resize_sig_ = nullptr;
  const cabana::Signal *hovered_sig_ = nullptr;
  std::set<BinaryIndex> selection_;
  ImVec2 grid_pos_;         // viewport origin of the current frame
  float column_width_ = 0;  // stretched section size
  bool under_mouse_ = false;
  bool scroll_to_top_ = false;
  Connections connections_;
};
