#include "tools/cabana/ui/tools/findsignal.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <thread>

#include "imgui.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/util.h"

// DoubleValidator (utils/qtutil.h): revert the edit when the new text is invalid
static bool doubleEdit(const char *label, std::string *text) {
  std::string prev = *text;
  bool changed = inputText(label, text);
  if (changed && validateDouble(*text) == ValidState::Invalid) {
    *text = prev;
    changed = false;
  }
  return changed;
}

// QString::toDouble: 0 when the text is not a valid number
static double toDouble(const std::string &s) {
  char *end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  return (end != s.c_str() && *end == '\0') ? v : 0.0;
}

// QString::toUShort / toULong(base): 0 when the text is not fully consumed
static unsigned long toULong(const std::string &s, int base = 10) {
  char *end = nullptr;
  unsigned long v = std::strtoul(s.c_str(), &end, base);
  return (end != s.c_str() && *end == '\0') ? v : 0;
}

static std::string trimmed(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (true) {
    size_t pos = s.find(sep, start);
    parts.push_back(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
    if (pos == std::string::npos) break;
    start = pos + 1;
  }
  return parts;
}

// FindSignalModel

std::string FindSignalModel::headerData(int section, bool horizontal) const {
  static std::string titles[] = {"Id", "Start Bit, size", "(time, value)"};
  return horizontal ? titles[section] : std::to_string(section + 1);
}

std::string FindSignalModel::data(int row, int column) const {
  const auto &s = filtered_signals[row];
  switch (column) {
    case 0: return s.id.toString();
    case 1: return std::to_string(s.sig.start_bit) + ", " + std::to_string(s.sig.size);
    case 2: {
      std::string joined;
      for (size_t i = 0; i < s.values.size(); ++i) {
        if (i) joined += " ";
        joined += s.values[i];
      }
      return joined;
    }
  }
  return {};
}

void FindSignalModel::search(std::function<bool(double)> cmp, const std::atomic<bool> &cancel) {
  std::mutex lock;
  const auto prev_sigs = !histories.empty() ? histories.back() : initial_signals;
  search_results.clear();
  search_results.reserve(prev_sigs.size());

  unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
  size_t chunk = (prev_sigs.size() + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < num_threads && t * chunk < (size_t)prev_sigs.size(); ++t) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, (size_t)prev_sigs.size());
    threads.emplace_back([&, start, end]() {
      for (size_t i = start; i < end && !cancel; ++i) {
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
          search_results.push_back({.id = s.id, .mono_time = (*it)->mono_time, .sig = s.sig, .values = values});
        }
      }
    });
  }
  for (auto &th : threads) th.join();
}

void FindSignalModel::applySearch() {
  filtered_signals = std::move(search_results);
  search_results.clear();
  histories.push_back(filtered_signals);
}

void FindSignalModel::undo() {
  if (!histories.empty()) {
    histories.pop_back();
    filtered_signals.clear();
    if (!histories.empty()) filtered_signals = histories.back();
  }
}

void FindSignalModel::reset() {
  histories.clear();
  filtered_signals.clear();
  initial_signals.clear();
}

// FindSignalDlg
FindSignalDlg::FindSignalDlg() {
  char buf[64];
  snprintf(buf, sizeof(buf), "Find Signal###findsignal%p", (void *)this);
  title_ = buf;
  model = std::make_unique<FindSignalModel>();
}

FindSignalDlg::~FindSignalDlg() {
  *alive_ = false;
  cancel_search_ = true;
  if (search_thread_.joinable()) search_thread_.join();
}

bool FindSignalDlg::draw() {
  if (!open_) return false;
  ImGui::SetNextWindowSize(ImVec2(900, 650), ImGuiCond_Appearing);
  setNextWindowFloatsOut();  // QDialog
  if (ImGui::Begin(title_.c_str(), &open_)) {
    // Messages group
    float group_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    ImGui::BeginChild("Messages", ImVec2(group_w, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    drawMessageGroup();
    ImGui::EndChild();
    ImGui::SameLine();
    // Signal group
    ImGui::BeginChild("Signal", ImVec2(group_w, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    drawPropertiesGroup();
    ImGui::EndChild();
    // find group
    float footer = stats_label_visible ? ImGui::GetTextLineHeightWithSpacing() : 0;
    ImGui::BeginChild("Find signal", ImVec2(0, -footer), ImGuiChildFlags_Borders);
    drawFindGroup();
    ImGui::EndChild();
    if (stats_label_visible) ImGui::TextUnformatted(stats_label.c_str());
    // QDialog closes on Escape
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Escape, false) &&
        !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
      open_ = false;
    }
  }
  ImGui::End();
  return open_;
}

void FindSignalDlg::drawMessageGroup() {
  ImGui::BeginDisabled(!message_group_enabled);
  ImGui::TextUnformatted("Messages");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Bus");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(-1);
  inputText("##bus", &bus_edit, "comma-separated values. Leave blank for all");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Address");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(-1);
  inputText("##address", &address_edit, "comma-separated hex values. Leave blank for all");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Time");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(70);
  doubleEdit("##first_time", &first_time_edit);
  ImGui::SameLine();
  ImGui::TextUnformatted("-");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70);
  doubleEdit("##last_time", &last_time_edit);
  ImGui::SameLine();
  ImGui::TextUnformatted("seconds");
  ImGui::EndDisabled();
}

void FindSignalDlg::drawPropertiesGroup() {
  ImGui::BeginDisabled(!properties_group_enabled);
  ImGui::TextUnformatted("Signal");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Size");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(70);
  if (ImGui::InputInt("##min_size", &min_size, 1, 10)) min_size = std::clamp(min_size, 1, 64);
  ImGui::SameLine();
  ImGui::TextUnformatted("-");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70);
  if (ImGui::InputInt("##max_size", &max_size, 1, 10)) max_size = std::clamp(max_size, 1, 64);
  ImGui::SameLine();
  checkBox("Little endian", &litter_endian);
  ImGui::SameLine();
  checkBox("Signed", &is_signed);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Factor");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(100);
  doubleEdit("##factor", &factor_edit);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Offset");
  ImGui::SameLine(80);
  ImGui::SetNextItemWidth(100);
  doubleEdit("##offset", &offset_edit);
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
  if (comboBox("##compare", &compare_cb, compare_items, compare_count)) {
    to_label_visible = compare_cb == compare_count - 1;
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();  // value1->setFocus
  doubleEdit("##value1", &value1);
  if (to_label_visible) {
    ImGui::SameLine();
    ImGui::TextUnformatted("-");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    doubleEdit("##value2", &value2);
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(!undo_btn_enabled || searching_);
  if (ImGui::Button("Undo prev find")) {
    model->undo();
    modelReset();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!search_btn_enabled);
  if (ImGui::Button(search_btn_text.c_str())) search();
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!reset_btn_enabled || searching_);
  if (ImGui::Button("Reset")) {
    model->reset();
    modelReset();
  }
  ImGui::EndDisabled();

  drawTable();
}

void FindSignalDlg::drawTable() {
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("view", model->columnCount() + 1, flags, ImVec2(0, 0))) return;
  ImGui::TableSetupScrollFreeze(0, 1);
  // vertical header: row number. QHeaderView has no width while the model is empty
  ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed | (model->rowCount() ? 0 : ImGuiTableColumnFlags_Disabled), 40.0f);
  for (int c = 0; c < model->columnCount(); ++c) {
    auto column_flags = c == model->columnCount() - 1 ? ImGuiTableColumnFlags_WidthStretch : ImGuiTableColumnFlags_WidthFixed;
    ImGui::TableSetupColumn(model->headerData(c, true).c_str(), column_flags, c == 0 ? 80.0f : 120.0f);
  }
  tableHeadersRow();
  for (int row = 0; row < model->rowCount(); ++row) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::PushID(row);
    if (ImGui::Selectable(model->headerData(row, false).c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) openMessage(model->filtered_signals[row].id);
    }
    customMenuRequested(row);
    ImGui::PopID();
    for (int c = 0; c < model->columnCount(); ++c) {
      ImGui::TableSetColumnIndex(c + 1);
      ImGui::TextUnformatted(model->data(row, c).c_str());
    }
  }
  ImGui::EndTable();
}

void FindSignalDlg::search() {
  if (model->histories.empty()) {
    setInitialSignals();
  }
  auto v1 = toDouble(value1);
  auto v2 = toDouble(value2);
  std::function<bool(double)> cmp = nullptr;
  switch (compare_cb) {
    case 0: cmp = [v1](double v) { return v == v1;}; break;
    case 1: cmp = [v1](double v) { return v > v1;}; break;
    case 2: cmp = [v1](double v) { return v >= v1;}; break;
    case 3: cmp = [v1](double v) { return v != v1;}; break;
    case 4: cmp = [v1](double v) { return v < v1;}; break;
    case 5: cmp = [v1](double v) { return v <= v1;}; break;
    case 6: cmp = [v1, v2](double v) { return v >= v1 && v <= v2;}; break;
  }
  properties_group_enabled = false;
  message_group_enabled = false;
  search_btn_enabled = false;
  stats_label_visible = false;
  search_btn_text = "Finding ....";
  // Qt runs model->search() from QTimer::singleShot(0) on the GUI thread (it joins its worker threads);
  // here it runs off the render thread into model->search_results and applies them with modelReset
  searching_ = true;
  if (search_thread_.joinable()) search_thread_.join();
  cancel_search_ = false;
  search_thread_ = std::thread([this, cmp, alive = std::weak_ptr<bool>(alive_)]() {
    model->search(cmp, cancel_search_);
    utils::runOnMainThread([this, alive]() {
      if (auto a = alive.lock(); a && *a) {
        searching_ = false;
        model->applySearch();
        modelReset();
      }
    });
  });
}

void FindSignalDlg::setInitialSignals() {
  std::set<unsigned short> buses;
  for (auto bus : split(trimmed(bus_edit), ',')) {
    bus = trimmed(bus);
    if (!bus.empty()) buses.insert((unsigned short)toULong(bus));
  }

  std::set<uint32_t> addresses;
  for (auto addr : split(trimmed(address_edit), ',')) {
    addr = trimmed(addr);
    if (!addr.empty()) addresses.insert(toULong(addr, 16));
  }

  cabana::Signal sig{};
  sig.is_little_endian = litter_endian;
  sig.is_signed = is_signed;
  sig.factor = toDouble(factor_edit);
  sig.offset = toDouble(offset_edit);

  double first_time_val = toDouble(first_time_edit);
  double last_time_val = toDouble(last_time_edit);
  auto [first_sec, last_sec] = std::minmax(first_time_val, last_time_val);
  uint64_t first_time = can->toMonoTime(first_sec);
  model->last_time = std::numeric_limits<uint64_t>::max();
  if (last_sec > 0) {
    model->last_time = can->toMonoTime(last_sec);
  }
  model->initial_signals.clear();

  for (const auto &[id, m] : can->lastMessages()) {
    if ((buses.empty() || buses.count(id.source)) && (addresses.empty() || addresses.count(id.address))) {
      const auto &events = can->events(id);
      auto e = std::lower_bound(events.cbegin(), events.cend(), first_time, CompareCanEvent());
      if (e != events.cend()) {
        const int total_size = m.dat.size() * 8;
        for (int size = min_size; size <= max_size; ++size) {
          for (int start = 0; start <= total_size - size; ++start) {
            FindSignalModel::SearchSignal s{.id = id, .mono_time = first_time, .sig = sig};
            s.sig.start_bit = start;
            s.sig.size = size;
            updateMsbLsb(s.sig);
            s.value = get_raw_value((*e)->dat, (*e)->size, s.sig);
            model->initial_signals.push_back(s);
          }
        }
      }
    }
  }
}

void FindSignalDlg::modelReset() {
  properties_group_enabled = model->histories.empty();
  message_group_enabled = model->histories.empty();
  search_btn_text = model->histories.empty() ? "Find" : "Find Next";
  reset_btn_enabled = !model->histories.empty();
  undo_btn_enabled = model->histories.size() > 1;
  search_btn_enabled = model->rowCount() > 0 || model->histories.empty();
  stats_label_visible = true;
  stats_label = std::to_string(model->filtered_signals.size()) + " matches. right click on an item to create signal. double click to open message";
}

void FindSignalDlg::customMenuRequested(int row) {
  if (ImGui::BeginPopupContextItem("menu")) {
    if (ImGui::MenuItem("Create Signal")) {
      auto &s = model->filtered_signals[row];
      UndoStack::instance()->push(new AddSigCommand(s.id, s.sig));
      openMessage(s.id);
    }
    ImGui::EndPopup();
  }
}
