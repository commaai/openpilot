#pragma once

#include <charconv>
#include <optional>
#include <set>

#include "json11/json11.hpp"
#include "tools/cabana/core/message_id.h"
#include "tools/cabana/core/color.h"
#include "tools/cabana/analysis/equations.h"
#include "tools/cabana/ui/chart/analysis.h"

namespace chart {
struct LayoutSignal {
  MessageId id;
  std::string name;
  TransformSettings transform;
  bool visible;
  std::string path;
  CabanaColor color{0, 114, 178};
};
struct LayoutChart {
  int type;
  std::vector<LayoutSignal> signals;
  std::string title;
  std::optional<double> y_min, y_max;
};
struct Layout {
  int columns;
  int range;
  std::vector<std::vector<LayoutChart>> tabs;
  std::vector<std::string> tab_names;
  std::vector<cabana::Equation> equations;
};

inline std::optional<Layout> parseLayout(const std::string &contents) {
  using json11::Json;
  std::string error;
  const auto doc = Json::parse(contents, error);
  auto integer = [](const Json &v, int min, int max) {
    return v.is_number() && v.number_value() >= min && v.number_value() <= max && v.number_value() == v.int_value();
  };
  if (!error.empty() || !integer(doc["cabana_layout"], 1, 2) || !integer(doc["columns"], 1, 4) ||
      !integer(doc["range"], 1, 86400) || !doc["tabs"].is_array() || doc["tabs"].array_items().empty()) return std::nullopt;
  if ((!doc["tab_names"].is_null() && !doc["tab_names"].is_array()) ||
      (!doc["equations"].is_null() && !doc["equations"].is_array())) return std::nullopt;
  Layout result{doc["columns"].int_value(), doc["range"].int_value(), {}};
  for (const auto &tab : doc["tabs"].array_items()) {
    if (!tab.is_array()) return std::nullopt;
    auto &charts = result.tabs.emplace_back();
    for (const auto &c : tab.array_items()) {
      if (!integer(c["type"], 0, 2) || !c["signals"].is_array()) return std::nullopt;
      auto &chart = charts.emplace_back(LayoutChart{c["type"].int_value(), {}});
      chart.title = c["title"].string_value();
      for (const auto &key : {"y_min", "y_max"}) {
        if (c[key].is_null()) continue;
        if (!c[key].is_number() || !std::isfinite(c[key].number_value())) return std::nullopt;
        (std::string(key) == "y_min" ? chart.y_min : chart.y_max) = c[key].number_value();
      }
      if (chart.y_min && chart.y_max && *chart.y_min >= *chart.y_max) return std::nullopt;
      for (const auto &s : c["signals"].array_items()) {
        if ((!s["message"].is_string() && !s["path"].is_string()) || !s["signal"].is_string() || s["signal"].string_value().empty() || !s["visible"].is_bool() ||
            !integer(s["transform"], 0, 3) || !integer(s["window"], 1, 100000) ||
            !s["scale"].is_number() || !std::isfinite(s["scale"].number_value()) ||
            !s["offset"].is_number() || !std::isfinite(s["offset"].number_value())) return std::nullopt;
        const std::string path = s["path"].string_value();
        if (s["path"].is_string() && path.empty()) return std::nullopt;
        uint32_t source = 0, address = 0;
        if (path.empty()) {
          const auto &text = s["message"].string_value();
          const auto colon = text.find(':');
          if (colon == std::string::npos) return std::nullopt;
          const char *begin = text.data(), *end = begin + text.size();
          auto bus = std::from_chars(begin, begin + colon, source);
          auto addr = std::from_chars(begin + colon + 1, end, address, 16);
          if (bus.ec != std::errc() || bus.ptr != begin + colon || source > 255 ||
              addr.ec != std::errc() || addr.ptr != end) return std::nullopt;
        }
        CabanaColor color{0, 114, 178};
        if (!s["color"].is_null()) {
          const auto &hex = s["color"].string_value();
          uint32_t rgb = 0;
          if (hex.size() != 7 || hex[0] != '#') return std::nullopt;
          auto parsed = std::from_chars(hex.data() + 1, hex.data() + 7, rgb, 16);
          if (parsed.ec != std::errc() || parsed.ptr != hex.data() + 7) return std::nullopt;
          color = CabanaColor(rgb >> 16, rgb >> 8, rgb);
        }
        chart.signals.push_back({{(uint8_t)source, address}, s["signal"].string_value(),
          {(Transform)s["transform"].int_value(), s["scale"].number_value(), s["offset"].number_value(), s["window"].int_value()},
          s["visible"].bool_value(), path, color});
      }
    }
  }
  for (const auto &name : doc["tab_names"].array_items()) {
    if (!name.is_string()) return std::nullopt;
    result.tab_names.push_back(name.string_value());
  }
  std::set<std::string> equation_names;
  for (const auto &e : doc["equations"].array_items()) {
    if (!e["name"].is_string() || e["name"].string_value().empty() || !e["source"].is_string() ||
        !e["globals"].is_string() || !e["function"].is_string() || !e["additional"].is_array() ||
        !equation_names.insert(e["name"].string_value()).second) return std::nullopt;
    cabana::Equation equation{e["name"].string_value(), e["source"].string_value(), e["globals"].string_value(), e["function"].string_value(), {}};
    for (const auto &source : e["additional"].array_items()) {
      if (!source.is_string()) return std::nullopt;
      equation.additional.push_back(source.string_value());
    }
    result.equations.push_back(std::move(equation));
  }
  return result;
}
}  // namespace chart
