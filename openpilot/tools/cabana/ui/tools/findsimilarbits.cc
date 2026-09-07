#include "tools/cabana/ui/tools/findsimilarbits.h"

#include <algorithm>
#include <unordered_map>

#include "imgui.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/util.h"

FindSimilarBitsDlg::FindSimilarBitsDlg() {
  setTitle("Find Similar Bits");

  for (int bus : can->sources) {
    bus_items_.push_back(bus);
  }
  updateMessages();
}

void FindSimilarBitsDlg::updateMessages() {
  msg_items_.clear();
  msg_names_.clear();
  for (auto &[address, msg] : dbc()->getMessages(busAt(src_bus_))) {
    msg_items_.push_back({msg.name, address});
  }
  std::sort(msg_items_.begin(), msg_items_.end(), [](auto &l, auto &r) { return l.first < r.first; });
  for (auto &[name, _] : msg_items_) msg_names_.push_back(name);
  msg_index_ = 0;
}

bool FindSimilarBitsDlg::draw() {
  if (begin(ImVec2(700, 500))) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Find From:");
    ImGui::SameLine(90);
    ImGui::TextUnformatted("Bus");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    if (comboBox("##src_bus", &src_bus_, bus_items_.data(), (int)bus_items_.size())) updateMessages();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    comboBox("##msg", &msg_index_, msg_names_);
    ImGui::SameLine();
    ImGui::TextUnformatted("Byte Index");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##byte_idx", &byte_idx_, 1, 10)) byte_idx_ = std::clamp(byte_idx_, 0, 63);
    ImGui::SameLine();
    ImGui::TextUnformatted("Bit Index");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##bit_idx", &bit_idx_, 1, 10)) bit_idx_ = std::clamp(bit_idx_, 0, 7);

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Find In:");
    ImGui::SameLine(90);
    ImGui::TextUnformatted("Bus");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    comboBox("##find_bus", &find_bus_, bus_items_.data(), (int)bus_items_.size());
    ImGui::SameLine();
    ImGui::TextUnformatted("Equal");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::Combo("##equal", &equal_, "Yes\0No\0");
    ImGui::SameLine();
    ImGui::TextUnformatted("Minimum Message Count");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##min_msgs", &min_msgs_, 1, 10)) min_msgs_ = std::max(min_msgs_, 0);
    ImGui::SameLine();
    if (ImGui::Button("Find")) find();

    drawTable();
  }
  return end();
}

void FindSimilarBitsDlg::drawTable() {
  // columns are set by find(); until then the table is an empty frame
  if (!table_has_columns_) {
    ImGui::BeginChild("table", ImVec2(0, 0), ImGuiChildFlags_Borders);
    ImGui::EndChild();
    return;
  }
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings;
  if (!ImGui::BeginTable("table", 7, flags, ImVec2(0, 0))) return;
  ImGui::TableSetupScrollFreeze(0, 1);
  static const char *headers[] = {"Address", "Byte Index", "Bit Index", "Mismatches", "Messages", "Mismatched (%)"};
  // the fixed widths are section sizes: imgui adds the cell padding on top of the column width
  const float padding = ImGui::GetStyle().CellPadding.x * 2;
  ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 40.0f - padding);  // vertical header: row number
  for (int c = 0; c < 6; ++c) {
    ImGui::TableSetupColumn(headers[c], c == 5 ? ImGuiTableColumnFlags_WidthStretch : ImGuiTableColumnFlags_WidthFixed,
                            100.0f - padding);
  }
  tableHeadersRow();
  ImGuiListClipper clipper;
  clipper.Begin((int)table_.size());
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      auto &m = table_[i];
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::PushID(i);
      if (ImGui::Selectable(std::to_string(i + 1).c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          openMessage(MessageId{.source = busAt(find_bus_), .address = m.address});
        }
      }
      ImGui::PopID();
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%x", m.address);
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
  const uint32_t selected_address = msg_index_ < (int)msg_items_.size() ? msg_items_[msg_index_].second : 0;
  table_ = calcBits(busAt(src_bus_), selected_address, byte_idx_, bit_idx_, busAt(find_bus_), equal_ == 0, min_msgs_);
  table_has_columns_ = true;
}

std::vector<FindSimilarBitsDlg::Mismatch> FindSimilarBitsDlg::calcBits(uint8_t bus, uint32_t selected_address, int byte_idx,
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

  std::vector<Mismatch> result;
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
