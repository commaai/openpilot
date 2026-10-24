#pragma once

#include <cstdint>
#include <optional>
#include <vector>

// Parsers for the Qt frontend's persisted QByteArray blobs (QWidget::saveGeometry,
// QSplitter::saveState, QHeaderView::saveState). Used once to migrate a Qt cabana.json
// to the imgui frontend's ini state. No Qt and no imgui.
// TODO: Delete this migration (qtstate.{h,cc}, migrateQtState in inistate.cc and the Qt
// byte-array fields in Settings) after users have had time to migrate to ui_state.
namespace qtstate {

constexpr int kMessageColumnCount = 7;

struct QtGeometry {
  int x, y, w, h;
  bool maximized;
};

struct QtSplitter {
  float ratio;
};

struct QtHeaderState {
  int sort_section;
  int sort_order;  // 0 = Qt::AscendingOrder, 1 = Qt::DescendingOrder
  bool sort_shown;
  int visual[kMessageColumnCount];
  int width[kMessageColumnCount];
  bool hidden[kMessageColumnCount];
};

std::optional<QtGeometry> parseQtGeometry(const std::vector<uint8_t> &data);
std::optional<QtSplitter> parseQtSplitter(const std::vector<uint8_t> &data);
std::optional<QtHeaderState> parseQtHeaderState(const std::vector<uint8_t> &data);

}  // namespace qtstate
