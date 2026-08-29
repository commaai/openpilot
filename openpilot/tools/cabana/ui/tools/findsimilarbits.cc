#include "tools/cabana/ui/tools/findsimilarbits.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "imgui.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/imgui_util.h"

// QIntValidator: optional sign followed by digits
static bool validInt(const std::string &s) {
  size_t i = (!s.empty() && (s[0] == '-' || s[0] == '+')) ? 1 : 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
  }
  return true;
}

// QString::toInt: 0 when the text is not a valid number
static int toInt(const std::string &s) {
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  return (end != s.c_str() && *end == '\0') ? (int)v : 0;
}

FindSimilarBitsDlg::FindSimilarBitsDlg() {
  char buf[64];
  snprintf(buf, sizeof(buf), "Find similar bits###findsimilarbits%p", (void *)this);
  title_ = buf;

  for (int bus : can->sources) {
    bus_items.push_back(bus);
  }

  // TODO: update when src_bus_combo changes
  for (auto &[address, msg] : dbc()->getMessages(-1)) {
    msg_items.push_back({msg.name, address});
  }
  std::sort(msg_items.begin(), msg_items.end(), [](auto &l, auto &r) { return l.first < r.first; });  // msg_cb->model()->sort(0)
  msg_cb = 0;
}

bool FindSimilarBitsDlg::draw() {
  if (!open_) return false;
  ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_Appearing);
  if (ImGui::Begin(title_.c_str(), &open_)) {
    std::string bus_labels;
    for (int bus : bus_items) bus_labels += std::to_string(bus) + '\0';

    // src_layout
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Find From:");
    ImGui::SameLine(90);
    ImGui::TextUnformatted("Bus");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::Combo("##src_bus", &src_bus_combo, bus_labels.c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##msg", msg_cb < (int)msg_items.size() ? msg_items[msg_cb].first.c_str() : "")) {
      for (int i = 0; i < (int)msg_items.size(); ++i) {
        ImGui::PushID(i);
        if (ImGui::Selectable(msg_items[i].first.c_str(), i == msg_cb)) msg_cb = i;
        ImGui::PopID();
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Byte Index");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##byte_idx", &byte_idx_sb, 1, 10)) byte_idx_sb = std::clamp(byte_idx_sb, 0, 63);  // byte_idx_sb->setRange(0, 63)
    ImGui::SameLine();
    ImGui::TextUnformatted("Bit Index");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##bit_idx", &bit_idx_sb, 1, 10)) bit_idx_sb = std::clamp(bit_idx_sb, 0, 7);  // bit_idx_sb->setRange(0, 7)

    // find_layout
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Find In:");
    ImGui::SameLine(90);
    ImGui::TextUnformatted("Bus");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::Combo("##find_bus", &find_bus_combo, bus_labels.c_str());
    ImGui::SameLine();
    ImGui::TextUnformatted("Equal");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::Combo("##equal", &equal_combo, "Yes\0No\0");
    ImGui::SameLine();
    ImGui::TextUnformatted("Min msg count");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    std::string prev = min_msgs;
    if (inputText("##min_msgs", &min_msgs) && !validInt(min_msgs)) {
      min_msgs = prev;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!search_btn_enabled);
    if (ImGui::Button("Find")) find();
    ImGui::EndDisabled();

    drawTable();
  }
  ImGui::End();
  return open_;
}

void FindSimilarBitsDlg::drawTable() {
  // columns are set by find(); the table is blank until then
  if (!table_has_columns) return;
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("table", 7, flags, ImVec2(0, 0))) return;
  ImGui::TableSetupScrollFreeze(0, 1);
  static const char *headers[] = {"address", "byte idx", "bit idx", "mismatches", "total msgs", "% mismatched"};
  ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 40.0f);  // vertical header: row number
  for (int c = 0; c < 6; ++c) {
    ImGui::TableSetupColumn(headers[c], c == 5 ? ImGuiTableColumnFlags_WidthStretch : ImGuiTableColumnFlags_WidthFixed, 100.0f);
  }
  ImGui::TableHeadersRow();
  ImGuiListClipper clipper;
  clipper.Begin((int)table.size());
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      auto &m = table[i];
      char address[32];
      snprintf(address, sizeof(address), "%x", m.address);
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::PushID(i);
      if (ImGui::Selectable(std::to_string(i + 1).c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          uint8_t find_bus = find_bus_combo < (int)bus_items.size() ? bus_items[find_bus_combo] : 0;
          MessageId msg_id = {.source = find_bus, .address = (uint32_t)std::strtoul(address, nullptr, 16)};
          openMessage(msg_id);
        }
      }
      ImGui::PopID();
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(address);
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%u", m.byte_idx);
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%u", m.bit_idx);
      ImGui::TableSetColumnIndex(4);
      ImGui::Text("%u", m.mismatches);
      ImGui::TableSetColumnIndex(5);
      ImGui::Text("%u", m.total);
      ImGui::TableSetColumnIndex(6);
      ImGui::Text("%.2f", m.perc);
    }
  }
  ImGui::EndTable();
}

void FindSimilarBitsDlg::find() {
  search_btn_enabled = false;
  table.clear();
  uint32_t selected_address = msg_cb < (int)msg_items.size() ? msg_items[msg_cb].second : 0;
  uint8_t src_bus = src_bus_combo < (int)bus_items.size() ? bus_items[src_bus_combo] : 0;
  uint8_t find_bus = find_bus_combo < (int)bus_items.size() ? bus_items[find_bus_combo] : 0;
  auto msg_mismatched = calcBits(src_bus, selected_address, byte_idx_sb, bit_idx_sb,
                                 find_bus, equal_combo == 0, toInt(min_msgs));
  table = std::move(msg_mismatched);
  table_has_columns = true;
  search_btn_enabled = true;
}

std::vector<FindSimilarBitsDlg::mismatched_struct> FindSimilarBitsDlg::calcBits(uint8_t bus, uint32_t selected_address, int byte_idx,
                                                                               int bit_idx, uint8_t find_bus, bool equal, int min_msgs_cnt) {
  std::unordered_map<uint32_t, std::vector<uint32_t>> mismatches;
  std::unordered_map<uint32_t, uint32_t> msg_count;
  const auto &events = can->allEvents();
  int bit_to_find = -1;
  for (const CanEvent *e : events) {
    if (e->src == bus) {
      if (e->address == selected_address && e->size > byte_idx) {
        bit_to_find = ((e->dat[byte_idx] >> (7 - bit_idx)) & 1) != 0;
      }
    }
    if (e->src == find_bus) {
      ++msg_count[e->address];
      if (bit_to_find == -1) continue;

      auto &mismatched = mismatches[e->address];
      if (mismatched.size() < e->size * 8) {
        mismatched.resize(e->size * 8);
      }
      for (int i = 0; i < e->size; ++i) {
        for (int j = 0; j < 8; ++j) {
          int bit = ((e->dat[i] >> (7 - j)) & 1) != 0;
          mismatched[i * 8 + j] += equal ? (bit != bit_to_find) : (bit == bit_to_find);
        }
      }
    }
  }

  std::vector<mismatched_struct> result;
  result.reserve(mismatches.size());
  for (auto it = mismatches.begin(); it != mismatches.end(); ++it) {
    if (auto cnt = msg_count[it->first]; cnt > (uint32_t)min_msgs_cnt) {
      auto &mismatched = it->second;
      for (int i = 0; i < (int)mismatched.size(); ++i) {
        if (float perc = (mismatched[i] / (double)cnt) * 100; perc < 50) {
          result.push_back({it->first, (uint32_t)i / 8, (uint32_t)i % 8, mismatched[i], cnt, perc});
        }
      }
    }
  }
  std::sort(result.begin(), result.end(), [](auto &l, auto &r) { return l.perc < r.perc; });
  return result;
}
