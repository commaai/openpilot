#include "tools/cabana/ui/widgets/map_widget.h"

#include <algorithm>
#include <cmath>

#include "imgui.h"
#include "implot.h"
#include "tools/cabana/streams/abstractstream.h"

namespace cabana {

MapWidget::MapWidget() = default;

void MapWidget::updatePoints() {
  const auto snapshot = can->cereal_series.snapshot();
  if (snapshot.revision == last_revision_) return;
  last_revision_ = snapshot.revision;

  times_.clear();
  lats_.clear();
  lons_.clear();
  bounds_set_ = false;

  const std::string lat_path = "/liveLocationKalman/positionGeodetic/value/0";
  const std::string lon_path = "/liveLocationKalman/positionGeodetic/value/1";
  const std::string alt_lat = "/gpsLocationExternal/latitude";
  const std::string alt_lon = "/gpsLocationExternal/longitude";

  std::string use_lat = lat_path, use_lon = lon_path;
  bool found = false;
  for (const auto &[seg, series] : snapshot.segments) {
    if (series->find(lat_path) != series->end()) {
      found = true;
      break;
    }
    if (series->find(alt_lat) != series->end()) {
      use_lat = alt_lat;
      use_lon = alt_lon;
      found = true;
      break;
    }
  }
  if (!found) return;

  std::vector<std::pair<double, double>> lat_pts;
  std::vector<std::pair<double, double>> lon_pts;
  for (const auto &[seg, series] : snapshot.segments) {
    auto it_lat = series->find(use_lat);
    auto it_lon = series->find(use_lon);
    if (it_lat != series->end()) {
      for (const auto &s : it_lat->second.samples) {
        const double sec = static_cast<double>(s.mono_time - std::min(s.mono_time, can->beginMonoTime())) / 1e9;
        lat_pts.emplace_back(sec, s.value);
      }
    }
    if (it_lon != series->end()) {
      for (const auto &s : it_lon->second.samples) {
        const double sec = static_cast<double>(s.mono_time - std::min(s.mono_time, can->beginMonoTime())) / 1e9;
        lon_pts.emplace_back(sec, s.value);
      }
    }
  }

  std::sort(lat_pts.begin(), lat_pts.end());
  std::sort(lon_pts.begin(), lon_pts.end());

  size_t i_lon = 0;
  for (const auto &[t, lat] : lat_pts) {
    while (i_lon + 1 < lon_pts.size() && lon_pts[i_lon + 1].first <= t) ++i_lon;
    if (i_lon < lon_pts.size() && std::abs(lon_pts[i_lon].first - t) < 0.1) {
      double lon = lon_pts[i_lon].second;
      if (std::abs(lat) > 0.0001 && std::abs(lon) > 0.0001) {
        times_.push_back(t);
        lats_.push_back(lat);
        lons_.push_back(lon);
        if (!bounds_set_) {
          min_lat_ = max_lat_ = lat;
          min_lon_ = max_lon_ = lon;
          bounds_set_ = true;
        } else {
          min_lat_ = std::min(min_lat_, lat);
          max_lat_ = std::max(max_lat_, lat);
          min_lon_ = std::min(min_lon_, lon);
          max_lon_ = std::max(max_lon_, lon);
        }
      }
    }
  }
}

void MapWidget::draw() {
  if (!visible) return;

  ImGui::SetNextWindowSize(ImVec2(400, 350), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Map###MapWidget", &visible)) {
    ImGui::End();
    return;
  }

  updatePoints();

  if (lats_.empty()) {
    ImGui::TextUnformatted("No GPS / localization coordinates available.");
    ImGui::End();
    return;
  }

  const double cur_sec = can->currentSec();
  double cur_lat = lats_.front(), cur_lon = lons_.front();
  auto it = std::upper_bound(times_.begin(), times_.end(), cur_sec);
  if (it != times_.begin()) {
    size_t idx = std::distance(times_.begin(), it - 1);
    cur_lat = lats_[idx];
    cur_lon = lons_[idx];
  }

  if (ImPlot::BeginPlot("##map_plot", ImVec2(-1, -1), ImPlotFlags_Equal)) {
    ImPlot::SetupAxes("Longitude", "Latitude");
    ImPlotSpec line_spec;
    line_spec.LineColor = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);
    line_spec.LineWeight = 2;
    ImPlot::PlotLine("Path", lons_.data(), lats_.data(), (int)lons_.size(), line_spec);

    ImPlotSpec dots;
    dots.LineColor = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
    dots.Marker = ImPlotMarker_Circle;
    dots.MarkerSize = 6;
    ImPlot::PlotScatter("Vehicle", &cur_lon, &cur_lat, 1, dots);
    ImPlot::EndPlot();
  }

  ImGui::End();
}

}  // namespace cabana
