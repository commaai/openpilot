#include "tools/cabana/ui/tools/findsignal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>

#include "imgui.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

namespace {
constexpr int MAX_ROWS = 300;
}  // namespace

void SignalSearch::search(const std::function<bool(double)> &cmp) {
  const auto prev_sigs = !histories.empty() ? histories.back() : initial_signals;
  filtered_signals.clear();
  filtered_signals.reserve(prev_sigs.size());

  std::mutex lock;
  parallelFor(prev_sigs.size(), [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const auto &s = prev_sigs[i];
      const auto &events = can->events(s.id);
      auto first = std::upper_bound(events.cbegin(), events.cend(), s.mono_time, CompareCanEvent());
      auto last = events.cend();
      if (last_time < std::numeric_limits<uint64_t>::max()) {
        last = std::upper_bound(events.cbegin(), events.cend(), last_time, CompareCanEvent());
      }

      auto it = std::find_if(first, last, [&](const CanEvent *e) { return cmp(get_raw_value(e->dat, e->size, s.sig)); });
      if (it != last) {
        auto values = s.values;
        char buf[64];
        snprintf(buf, sizeof(buf), "(%.3f, %g)", can->toSeconds((*it)->mono_time), get_raw_value((*it)->dat, (*it)->size, s.sig));
        values.push_back(buf);
        std::lock_guard lk(lock);
        filtered_signals.push_back({.id = s.id, .mono_time = (*it)->mono_time, .sig = s.sig, .values = values});
      }
    }
  });

  histories.push_back(filtered_signals);
}

void SignalSearch::undo() {
  if (!histories.empty()) {
    histories.pop_back();
    filtered_signals.clear();
    if (!histories.empty()) filtered_signals = histories.back();
  }
}

void SignalSearch::reset() {
  histories.clear();
  filtered_signals.clear();
  initial_signals.clear();
}

FindSignalDlg::FindSignalDlg() {
  setTitle("Find Signal");
}

FindSignalDlg::~FindSignalDlg() {
  if (search_future_.valid()) search_future_.wait();
}

bool FindSignalDlg::draw() {
  if (search_future_.valid() && search_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    search_future_.get();
    searched_ = true;
  }
  searching_ = search_future_.valid();
  if (begin(ImVec2(900, 650))) {
    float group_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    ImGui::BeginChild("Messages", ImVec2(group_w, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    drawMessageGroup();
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("Signal", ImVec2(group_w, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    drawPropertiesGroup();
    ImGui::EndChild();
    float footer = searched_ ? ImGui::GetTextLineHeightWithSpacing() : 0;
    ImGui::BeginChild("Find signal", ImVec2(0, -footer), ImGuiChildFlags_Borders);
    drawFindGroup();
    ImGui::EndChild();
    if (searched_) {
      ImGui::Text("%zu matches. right click on an item to create signal. double click to open message",
                  search_.filtered_signals.size());
    }
  }
  return end();
}

void FindSignalDlg::drawMessageGroup() {
  ImGui::BeginDisabled(searching_ || !search_.histories.empty());
  ImGui::TextUnformatted("Messages");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Bus");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(-1);
  inputText("##bus", &bus_, "comma-separated values. Leave blank for all");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Address");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(-1);
  inputText("##address", &address_, "comma-separated hex values. Leave blank for all");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Time");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(70);
  validatedText("##first_time", &first_time_, validateDouble);
  ImGui::SameLine();
  ImGui::TextUnformatted("-");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70);
  validatedText("##last_time", &last_time_, validateDouble);
  ImGui::SameLine();
  ImGui::TextUnformatted("seconds");
  ImGui::EndDisabled();
}

void FindSignalDlg::drawPropertiesGroup() {
  ImGui::BeginDisabled(searching_ || !search_.histories.empty());
  ImGui::TextUnformatted("Signal");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Size");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(70);
  if (ImGui::InputInt("##min_size", &min_size_, 1, 10)) min_size_ = std::clamp(min_size_, 1, 64);
  ImGui::SameLine();
  ImGui::TextUnformatted("-");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70);
  if (ImGui::InputInt("##max_size", &max_size_, 1, 10)) max_size_ = std::clamp(max_size_, 1, 64);
  ImGui::SameLine();
  checkBox("Little endian", &little_endian_);
  ImGui::SameLine();
  checkBox("Signed", &is_signed_);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Factor");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(100);
  validatedText("##factor", &factor_, validateDouble);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Offset");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(100);
  validatedText("##offset", &offset_, validateDouble);
  ImGui::EndDisabled();
}

void FindSignalDlg::drawFindGroup() {
  static const char *compare_items[] = {"=", ">", ">=", "!=", "<", "<=", "between"};
  const int compare_count = IM_ARRAYSIZE(compare_items);
  ImGui::TextUnformatted("Find signal");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Value");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(90);
  ImGui::Combo("##compare", &compare_, compare_items, compare_count);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
  validatedText("##value1", &value1_, validateDouble);
  if (compare_ == compare_count - 1) {
    ImGui::SameLine();
    ImGui::TextUnformatted("-");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    validatedText("##value2", &value2_, validateDouble);
  }
  ImGui::SameLine();
  const bool first = !searching_ && search_.histories.empty();
  ImGui::BeginDisabled(searching_ || search_.histories.size() <= 1);
  if (ImGui::Button("Undo prev find")) {
    search_.undo();
    searched_ = true;
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(searching_ || (search_.filtered_signals.empty() && !first));
  if (ImGui::Button(searching_ ? "Finding ...." : (first ? "Find" : "Find Next"))) search();
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(searching_ || first);
  if (ImGui::Button("Reset")) {
    search_.reset();
    searched_ = true;
  }
  ImGui::EndDisabled();

  if (searching_) {
    ImGui::BeginChild("view", ImVec2(0, 0), ImGuiChildFlags_Borders);
    ImGui::EndChild();
  } else {
    drawTable();
  }
}

void FindSignalDlg::drawTable() {
  static const char *titles[] = {"Id", "Start Bit, size", "(time, value)"};
  const int columns = IM_ARRAYSIZE(titles);
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings;
  if (!ImGui::BeginTable("view", columns + 1, flags, ImVec2(0, 0))) return;
  ImGui::TableSetupScrollFreeze(0, 1);
  const int rows = std::min<int>(search_.filtered_signals.size(), MAX_ROWS);
  // vertical header: row number, no width while there are no results
  ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed | (rows ? 0 : ImGuiTableColumnFlags_Disabled), 40.0f);
  for (int c = 0; c < columns; ++c) {
    auto column_flags = c == columns - 1 ? ImGuiTableColumnFlags_WidthStretch : ImGuiTableColumnFlags_WidthFixed;
    ImGui::TableSetupColumn(titles[c], column_flags, c == 0 ? 80.0f : 120.0f);
  }
  tableHeadersRow();
  for (int row = 0; row < rows; ++row) {
    const auto &s = search_.filtered_signals[row];
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::PushID(row);
    if (ImGui::Selectable(std::to_string(row + 1).c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) openMessage(s.id);
    }
    drawContextMenu(row);
    ImGui::PopID();
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(s.id.toString().c_str());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("%d, %d", s.sig.start_bit, s.sig.size);
    ImGui::TableSetColumnIndex(3);
    std::string values;
    for (size_t i = 0; i < s.values.size(); ++i) {
      if (i) values += " ";
      values += s.values[i];
    }
    ImGui::TextUnformatted(values.c_str());
  }
  ImGui::EndTable();
}

void FindSignalDlg::search() {
  if (search_.histories.empty()) {
    setInitialSignals();
  }
  auto v1 = utils::toDouble(value1_);
  auto v2 = utils::toDouble(value2_);
  std::function<bool(double)> cmp = nullptr;
  switch (compare_) {
    case 0: cmp = [v1](double v) { return v == v1;}; break;
    case 1: cmp = [v1](double v) { return v > v1;}; break;
    case 2: cmp = [v1](double v) { return v >= v1;}; break;
    case 3: cmp = [v1](double v) { return v != v1;}; break;
    case 4: cmp = [v1](double v) { return v < v1;}; break;
    case 5: cmp = [v1](double v) { return v <= v1;}; break;
    case 6: cmp = [v1, v2](double v) { return v >= v1 && v <= v2;}; break;
  }
  searched_ = false;
  // a thread of its own: the search fans out over the pool, which a pool thread must not wait on
  search_future_ = std::async(std::launch::async, [this, cmp = std::move(cmp)]() { search_.search(cmp); });
  searching_ = true;
}

void FindSignalDlg::setInitialSignals() {
  std::set<unsigned short> buses;
  for (auto bus : utils::split(utils::trimmed(bus_), ',')) {
    bus = utils::trimmed(bus);
    if (!bus.empty()) buses.insert((unsigned short)utils::toULong(bus));
  }

  std::set<uint32_t> addresses;
  for (auto addr : utils::split(utils::trimmed(address_), ',')) {
    addr = utils::trimmed(addr);
    if (!addr.empty()) addresses.insert(utils::toULong(addr, 16));
  }

  cabana::Signal sig{};
  sig.is_little_endian = little_endian_;
  sig.is_signed = is_signed_;
  sig.factor = utils::toDouble(factor_);
  sig.offset = utils::toDouble(offset_);

  double first_time_val = utils::toDouble(first_time_);
  double last_time_val = utils::toDouble(last_time_);
  auto [first_sec, last_sec] = std::minmax(first_time_val, last_time_val);
  uint64_t first_time = can->toMonoTime(first_sec);
  search_.last_time = std::numeric_limits<uint64_t>::max();
  if (last_sec > 0) {
    search_.last_time = can->toMonoTime(last_sec);
  }
  search_.initial_signals.clear();

  for (const auto &[id, m] : can->lastMessages()) {
    if ((buses.empty() || buses.count(id.source)) && (addresses.empty() || addresses.count(id.address))) {
      const auto &events = can->events(id);
      auto e = std::lower_bound(events.cbegin(), events.cend(), first_time, CompareCanEvent());
      if (e != events.cend()) {
        const int total_size = m.dat.size() * 8;
        for (int size = min_size_; size <= max_size_; ++size) {
          for (int start = 0; start <= total_size - size; ++start) {
            SignalSearch::SearchSignal s{.id = id, .mono_time = first_time, .sig = sig};
            s.sig.start_bit = start;
            s.sig.size = size;
            updateMsbLsb(s.sig);
            s.value = get_raw_value((*e)->dat, (*e)->size, s.sig);
            search_.initial_signals.push_back(s);
          }
        }
      }
    }
  }
}

void FindSignalDlg::drawContextMenu(int row) {
  if (ImGui::BeginPopupContextItem("menu")) {
    if (ImGui::MenuItem("Create Signal")) {
      auto &s = search_.filtered_signals[row];
      UndoStack::instance()->push(new AddSigCommand(s.id, s.sig));
      openMessage(s.id);
    }
    ImGui::EndPopup();
  }
}
