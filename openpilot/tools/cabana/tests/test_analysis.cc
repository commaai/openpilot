#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "common/tests/native_test.h"
#include "tools/cabana/core/analysis.h"
#include "tools/cabana/ui/chart/downsample.h"

#ifndef CABANA_REPO_ROOT
#define CABANA_REPO_ROOT "."
#endif

template <typename F>
void requireError(F function, const std::string &part) {
  try { function(); }
  catch (const std::exception &error) {
    REQUIRE(std::string(error.what()).find(part) != std::string::npos);
    return;
  }
  throw std::runtime_error("Expected error containing: " + part);
}

void test_layouts() {
  const std::string input = R"({"cabana_layout":1,"current_tab_index":1,"tabs":[
    {"name":"Empty","charts":[]},
    {"name":"Analysis","charts":[{"title":"Two transforms","series_type":2,"y_limits":{"min":-2,"max":8},"curves":[
      {"name":"/carState/vEgo","color":"#11223344","visible":false,"transform":"derivative","derivative_dt":0.05,"scale":2,"offset":-3},
      {"name":"/carState/vEgo","scale":3},
      {"name":"STEER","can_id":"0:123","color":"#ffffff"},
      {"name":"error","custom_python":{"linked_source":"/carState/vEgo","additional_sources":["/controlsState/curvature"],"globals_code":"x=1","function_code":"return value-v1"}}
    ]}]}]})";
  auto layout = cabana::parseAnalysisLayout(input);
  REQUIRE(layout.current_tab == 1 && layout.tabs[0].charts.empty());
  const auto saved = cabana::writeAnalysisLayout(layout);
  auto restored = cabana::parseAnalysisLayout(saved);
  REQUIRE(cabana::writeAnalysisLayout(restored) == saved);
  const auto &chart = restored.tabs[1].charts[0];
  REQUIRE(chart.title == "Two transforms" && chart.series_type == 2);
  REQUIRE(chart.y_limits->first == -2 && chart.y_limits->second == 8);
  REQUIRE(chart.curves.size() == 4 && !chart.curves[0].visible);
  REQUIRE(chart.curves[0].transform.derivative && chart.curves[0].transform.derivative_dt == 0.05);
  REQUIRE(chart.curves[0].color == "#11223344" && chart.curves[0].transform.offset == -3);
  REQUIRE(chart.curves[3].python->linked_source == "carState/vEgo");
  REQUIRE(chart.curves[3].python->additional_sources[0] == "controlsState/curvature");
  requireError([] { cabana::parseAnalysisLayout("bad json"); }, "JSON");
  requireError([] { cabana::parseAnalysisLayout(R"({"tabs":[]})"); }, "no tabs");
  requireError([] { cabana::parseAnalysisLayout(R"({"cabana_layout":9,"tabs":[{}]})"); }, "version");
  requireError([] { cabana::parseAnalysisLayout(R"({"tabs":[{"root":{"kind":"camera"}}]})"); }, "camera pane");
  requireError([] { cabana::parseAnalysisLayout(R"({"tabs":[{"charts":[{"curves":[{"name":"unresolved"}]}]}]})"); }, "Unresolved");
  requireError([] { cabana::parseAnalysisLayout(R"({"tabs":[{"charts":[{"curves":[{"name":"/x","color":"#nope00"}]}]}]})"); }, "color");
  requireError([] { cabana::parseAnalysisLayout(R"({"tabs":[{"charts":[{"curves":[{"name":"/x","transform":"mystery"}]}]}]})"); }, "Unsupported transform");

  size_t count = 0, curves = 0;
  const auto directory = std::filesystem::path(CABANA_REPO_ROOT) / "openpilot/tools/jotpluggler/layouts";
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() != ".json") continue;
    const auto name = entry.path().stem().string();
    // Those layouts require non-plot panes or have no initialized pane.
    if (name == "cameras-and-map" || name == "driver-monitoring-debug" || name == "new-layout") continue;
    std::ifstream file(entry.path());
    std::stringstream text;
    text << file.rdbuf();
    auto imported = cabana::parseAnalysisLayout(text.str());
    REQUIRE(cabana::writeAnalysisLayout(cabana::parseAnalysisLayout(cabana::writeAnalysisLayout(imported))) ==
            cabana::writeAnalysisLayout(imported));
    for (const auto &tab : imported.tabs) for (const auto &plot : tab.charts) curves += plot.curves.size();
    ++count;
  }
  REQUIRE(count == 14 && curves > 200);
  std::cout << "Layout round trips: " << count << " layouts, " << curves << " curves\n";
}

void test_python() {
  using cabana::CerealPoints;
  const std::map<std::string, CerealPoints> inputs{
    {"reference", {{1000000000, 2}, {2000000000, 4}, {3000000000, 6}, {4000000000, 8}}},
    {"aux", {{1500000000, 10}, {3500000000, 20}}}
  };
  cabana::PythonSeries spec{"reference", {"aux"}, "", "return value + v1"};
  auto result = cabana::evaluatePythonSeries(spec, inputs, {});
  REQUIRE(result.size() == 3);
  REQUIRE(result[0].mono_time == 2000000000 && result[0].value == 14);
  REQUIRE(result[1].value == 16 && result[2].value == 28);
  spec.function_code = "return time";
  result = cabana::evaluatePythonSeries(spec, inputs, {}, 15, 1000000000);
  REQUIRE(result[0].mono_time == 2000000000 && result[0].value == 1);
  spec.globals_code = "total = 0";
  spec.function_code = "global total\nout=np.empty_like(value)\nfor i in range(len(value)):\n total += value[i]\n out[i]=total\nreturn out";
  result = cabana::evaluatePythonSeries(spec, inputs, {});
  REQUIRE(result[0].value == 4 && result[1].value == 10 && result[2].value == 18);
  REQUIRE(cabana::evaluatePythonSeries(spec, inputs, {})[0].value == 4);  // reset initialization on a new window
  spec.globals_code.clear();
  spec.function_code = "raise ValueError('visible failure')";
  requireError([&] { cabana::evaluatePythonSeries(spec, inputs, {}); }, "visible failure");
  spec.function_code = "return np.array([1])";
  requireError([&] { cabana::evaluatePythonSeries(spec, inputs, {}); }, "same shape");
  spec.function_code = "return (time[::-1], value)";
  requireError([&] { cabana::evaluatePythonSeries(spec, inputs, {}); }, "unsorted");
  spec.function_code = "return value";
  requireError([&] { cabana::evaluatePythonSeries(spec, {}, {}); }, "Waiting for Python input");
  spec.function_code = "while True:\n pass";
  auto start = std::chrono::steady_clock::now();
  requireError([&] { cabana::evaluatePythonSeries(spec, inputs, {}, 0.15); }, "timed out");
  REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::seconds(3));
  start = std::chrono::steady_clock::now();
  requireError([&] {
    cabana::evaluatePythonSeries(spec, inputs, [&] { return std::chrono::steady_clock::now() - start > std::chrono::milliseconds(80); });
  }, "cancelled");
  REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::seconds(3));
}

void test_envelope() {
  struct Point { double x, y; };
  std::vector<Point> points;
  for (int i = 0; i < 10000; ++i) points.push_back({i / 10000.0, 0});
  points[111].y = -17;
  points[112].y = 23;
  points[1000].y = std::numeric_limits<double>::quiet_NaN();
  auto reduced = chart::pixelEnvelope(points.begin(), points.end(), 0, 1, 100);
  REQUIRE(reduced.size() <= 410);
  REQUIRE(reduced.front().x == points.front().x && reduced.back().x == points.back().x);
  bool low = false, high = false, gap = false;
  double previous = -1;
  for (const auto &point : reduced) {
    REQUIRE(point.x >= previous);
    previous = point.x;
    low |= point.y == -17;
    high |= point.y == 23;
    gap |= std::isnan(point.y);
  }
  REQUIRE(low && high && gap);
}

int main() {
  return run_native_test([] { test_layouts(); test_python(); test_envelope(); std::cout << "Analysis tests passed\n"; });
}
