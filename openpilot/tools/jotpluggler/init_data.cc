#include "tools/jotpluggler/app.h"

#include <capnp/dynamic.h>

namespace {

std::string format_data(capnp::Data::Reader data) {
  const bool text = std::all_of(data.begin(), data.end(), [](uint8_t c) {
    return (c >= 32 && c < 127) || c == '\n' || c == '\r' || c == '\t';
  });
  if (data.size() == 0) return "";
  if (text) return std::string(reinterpret_cast<const char *>(data.begin()), data.size());
  constexpr char hex[] = "0123456789abcdef";
  std::string value = "hex: ";
  for (uint8_t c : data) {
    value += hex[c >> 4];
    value += hex[c & 15];
  }
  return value;
}

std::string format_value(capnp::DynamicValue::Reader value) {
  if (value.getType() == capnp::DynamicValue::DATA) return format_data(value.as<capnp::Data>());
  if (value.getType() == capnp::DynamicValue::TEXT) {
    const auto text = value.as<capnp::Text>();
    return std::string(text.begin(), text.size());
  }
  return kj::str(value).cStr();
}

}  // namespace

InitDataSnapshot extract_init_data(cereal::InitData::Reader reader) {
  InitDataSnapshot snapshot;
  InitDataSection metadata{"Device & software", {}};
  auto dynamic = capnp::toDynamic(reader);
  for (auto field : dynamic.getSchema().getFields()) {
    const std::string name = field.getProto().getName().cStr();
    if (name == "deprecated") continue;
    if (name == "params" || name == "commands") {
      InitDataSection section{name, {}};
      auto entries = dynamic.get(field).as<capnp::DynamicStruct>().get("entries").as<capnp::DynamicList>();
      for (auto entry : entries) {
        auto item = entry.as<capnp::DynamicStruct>();
        section.values.emplace_back(format_value(item.get("key")), format_value(item.get("value")));
      }
      std::sort(section.values.begin(), section.values.end());
      snapshot.sections.push_back(std::move(section));
    } else {
      metadata.values.emplace_back(name, format_value(dynamic.get(field)));
    }
  }
  snapshot.sections.insert(snapshot.sections.begin(), std::move(metadata));
  return snapshot;
}

void draw_init_data_tab(AppSession *session, UiState *state) {
  ImGui::SetNextItemWidth(280.0f);
  input_text_with_hint_string("##init_data_search", "Search fields and values...", &state->init_data_search);
  ImGui::SameLine();
  ImGui::TextDisabled("Right-click a value to copy");
  ImGui::Separator();
  if (!session->route_data.init_data) {
    const auto load = session->route_loader ? session->route_loader->snapshot() : RouteLoadSnapshot{};
    ImGui::TextWrapped("%s", load.active ? "Loading initData..." : "No initData available for this route or stream.");
    return;
  }
  const std::string query = lowercase_copy(state->init_data_search);
  const auto matches = [&](const auto &row) {
    return query.empty() || lowercase_copy(row.first).find(query) != std::string::npos
      || lowercase_copy(row.second).find(query) != std::string::npos;
  };
  bool found = false;
  if (ImGui::BeginChild("##init_data_content", ImVec2(0, 0), false)) {
    for (const auto &section : session->route_data.init_data->sections) {
      const size_t count = std::count_if(section.values.begin(), section.values.end(), matches);
      if (!query.empty() && count == 0) continue;
      found = found || count > 0;
      ImGui::PushID(section.name.c_str());
      const std::string label = section.name + " (" + std::to_string(count) + ")";
      if (ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        if (section.values.empty()) ImGui::TextDisabled("No entries");
        if (ImGui::BeginTable("##values", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable)) {
          ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 230.0f);
          ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableHeadersRow();
          for (size_t i = 0; i < section.values.size(); ++i) {
            const auto &[key, value] = section.values[i];
            if (!matches(section.values[i])) continue;
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(key.c_str());
            ImGui::TableSetColumnIndex(1);
            app_push_mono_font();
            ImGui::TextWrapped("%s", value.empty() ? "(empty)" : value.c_str());
            app_pop_mono_font();
            if (ImGui::BeginPopupContextItem("##copy")) {
              if (ImGui::MenuItem("Copy value")) ImGui::SetClipboardText(value.c_str());
              if (ImGui::MenuItem("Copy field")) ImGui::SetClipboardText(key.c_str());
              ImGui::EndPopup();
            }
            ImGui::PopID();
          }
          ImGui::EndTable();
        }
      }
      ImGui::PopID();
    }
    if (!query.empty() && !found) ImGui::TextDisabled("No matching fields or values.");
  }
  ImGui::EndChild();
}
