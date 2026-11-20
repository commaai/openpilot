#include "tools/cabana/ui/qtstate.h"

#include <algorithm>
#include <map>

namespace qtstate {

namespace {

// big-endian QDataStream reader; any read past the end latches the failed state
class Cursor {
public:
  explicit Cursor(const std::vector<uint8_t> &data) : data_(data) {}

  bool ok() const { return ok_; }
  size_t remaining() const { return ok_ ? data_.size() - pos_ : 0; }

  uint8_t u8() { return read(1) ? data_[pos_ - 1] : 0; }
  uint16_t u16() {
    if (!read(2)) return 0;
    return (uint16_t(data_[pos_ - 2]) << 8) | data_[pos_ - 1];
  }
  uint32_t u32() {
    if (!read(4)) return 0;
    return (uint32_t(data_[pos_ - 4]) << 24) | (uint32_t(data_[pos_ - 3]) << 16) |
           (uint32_t(data_[pos_ - 2]) << 8) | data_[pos_ - 1];
  }
  int32_t i32() { return (int32_t)u32(); }
  void skip(size_t n) { read(n); }

private:
  bool read(size_t n) {
    if (!ok_ || data_.size() - pos_ < n) {
      ok_ = false;
      return false;
    }
    pos_ += n;
    return true;
  }

  const std::vector<uint8_t> &data_;
  size_t pos_ = 0;
  bool ok_ = true;
};

}  // namespace

std::optional<QtGeometry> parseQtGeometry(const std::vector<uint8_t> &data) {
  Cursor c(data);
  if (c.u32() != 0x01D9D0CB) return std::nullopt;
  const uint16_t major = c.u16();
  c.u16();  // minor
  if (!c.ok() || major > 3) return std::nullopt;

  c.skip(16);  // frameGeometry
  int x1 = c.i32(), y1 = c.i32(), x2 = c.i32(), y2 = c.i32();  // normalGeometry
  c.i32();  // screenNumber
  const bool maximized = c.u8() != 0;
  c.u8();  // fullScreen
  if (major >= 2) c.i32();  // screenWidth
  if (major >= 3) {
    // the client-area rect actually restored
    x1 = c.i32(); y1 = c.i32(); x2 = c.i32(); y2 = c.i32();
  }
  if (!c.ok()) return std::nullopt;
  return QtGeometry{x1, y1, x2 - x1 + 1, y2 - y1 + 1, maximized};
}

std::optional<QtSplitter> parseQtSplitter(const std::vector<uint8_t> &data) {
  Cursor c(data);
  if (c.i32() != 0xff) return std::nullopt;
  const int32_t version = c.i32();
  if (!c.ok() || version > 1) return std::nullopt;

  const uint32_t count = c.u32();
  if (!c.ok() || count != 2) return std::nullopt;
  const int32_t first = c.i32();
  const int32_t second = c.i32();
  if (!c.ok() || first + second <= 0) return std::nullopt;
  return QtSplitter{first / float(first + second)};
}

std::optional<QtHeaderState> parseQtHeaderState(const std::vector<uint8_t> &data) {
  constexpr int N = kMessageColumnCount;
  Cursor c(data);
  if (c.i32() != 0xff) return std::nullopt;
  if (c.i32() != 0) return std::nullopt;  // version

  c.i32();  // orientation
  const int32_t sort_order = c.i32();
  const int32_t sort_section = c.i32();
  const bool sort_shown = c.u8() != 0;
  if (!c.ok()) return std::nullopt;

  QtHeaderState state{};
  state.sort_section = sort_section;
  state.sort_order = sort_order;
  state.sort_shown = sort_shown;
  for (int i = 0; i < N; ++i) state.visual[i] = i;

  // visualIndices: logical -> visual, empty means identity
  const uint32_t visual_count = c.u32();
  if (!c.ok() || visual_count > c.remaining() / 4) return std::nullopt;
  if (visual_count > 0) {
    std::vector<int> visual(visual_count);
    for (uint32_t i = 0; i < visual_count; ++i) visual[i] = c.i32();
    if (!c.ok()) return std::nullopt;
    bool valid = visual_count == (uint32_t)N;
    if (valid) {
      bool seen[N] = {};
      for (int v : visual) {
        if (v < 0 || v >= N || seen[v]) { valid = false; break; }
        seen[v] = true;
      }
    }
    if (valid) {
      for (int i = 0; i < N; ++i) state.visual[i] = visual[i];
    }
  }

  // logicalIndices: visual -> logical, unused but must be consumed
  const uint32_t logical_count = c.u32();
  if (!c.ok() || logical_count > c.remaining() / 4) return std::nullopt;
  c.skip(logical_count * 4);

  // sectionHidden: QBitArray indexed by visual position
  const uint32_t hidden_bits = c.u32();
  if (!c.ok() || hidden_bits / 8 > c.remaining()) return std::nullopt;
  std::vector<uint8_t> hidden_bytes((hidden_bits + 7) / 8);
  for (auto &b : hidden_bytes) b = c.u8();
  if (!c.ok()) return std::nullopt;

  // hiddenSectionSize: logical index -> size before hiding
  const uint32_t hidden_size_count = c.u32();
  if (!c.ok() || hidden_size_count > c.remaining() / 8) return std::nullopt;
  std::map<int, int> hidden_sizes;
  for (uint32_t i = 0; i < hidden_size_count; ++i) {
    const int key = c.i32();
    hidden_sizes[key] = c.i32();
  }
  if (!c.ok()) return std::nullopt;

  c.i32();  // length
  c.i32();  // sectionCount
  c.skip(5);   // movable, clickable, highlight, stretchLastSection, cascading
  c.skip(24);  // stretchSections, contentsSections, defaultSectionSize, minimumSectionSize, defaultAlignment, globalResizeMode
  if (!c.ok()) return std::nullopt;

  // SectionItems in visual order; a Qt4 item with count != 1 stands for count sections
  const uint32_t item_count = c.u32();
  if (!c.ok() || item_count > c.remaining() / 12) return std::nullopt;
  std::vector<int> sizes;
  for (uint32_t i = 0; i < item_count; ++i) {
    const int size = c.i32();
    const int count = c.i32();
    c.i32();  // resizeMode
    if (!c.ok() || count <= 0 || (int)sizes.size() + count > N) return std::nullopt;
    for (int j = 0; j < count; ++j) sizes.push_back(size / count);
  }
  if (!c.ok() || (int)sizes.size() != N) return std::nullopt;

  for (int i = 0; i < N; ++i) {
    const int v = state.visual[i];
    state.hidden[i] = (uint32_t)v < hidden_bits && (hidden_bytes[v / 8] & (1 << (v % 8))) != 0;
    auto it = hidden_sizes.find(i);
    state.width[i] = (state.hidden[i] && it != hidden_sizes.end()) ? it->second : sizes[v];
  }
  return state;
}

}  // namespace qtstate
