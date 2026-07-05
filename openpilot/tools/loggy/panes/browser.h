#pragma once

#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/pane.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr const char *kLoggySeriesPathPayload = "LOGGY_SERIES_PATH";

struct BrowserState {
  std::string filter;
  size_t max_rows = 1000;
};

struct BrowserSeriesRow {
  std::string path;
  std::string label;
};

inline BrowserState parse_browser_state(std::string_view state_json) {
  BrowserState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["max_rows"].is_number()) {
    state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 10000));
  }
  return state;
}

inline std::string browser_state_json(const BrowserState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"max_rows", static_cast<int>(state.max_rows)},
  }).dump();
}

inline std::string browser_leaf_label(std::string_view path) {
  if (path.empty()) return "series";
  const size_t slash = path.find_last_of('/');
  const std::string_view label = slash == std::string_view::npos ? path : path.substr(slash + 1);
  return label.empty() ? std::string(path) : std::string(label);
}

inline std::string browser_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline bool browser_path_matches_filter(std::string_view path, std::string_view filter) {
  if (filter.empty()) return true;
  const std::string haystack = browser_lower_text(path);
  const std::string needle = browser_lower_text(filter);
  return haystack.find(needle) != std::string::npos;
}

inline std::vector<BrowserSeriesRow> prepare_browser_series_rows(const Store &store, const BrowserState &state) {
  const std::vector<std::string> paths = store.seriesPaths();
  std::vector<BrowserSeriesRow> rows;
  rows.reserve(std::min(paths.size(), state.max_rows));
  for (const std::string &path : paths) {
    if (!browser_path_matches_filter(path, state.filter)) continue;
    rows.push_back({path, browser_leaf_label(path)});
    if (rows.size() >= state.max_rows) break;
  }
  return rows;
}

void draw_browser_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
