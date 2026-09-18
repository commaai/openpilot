#include <filesystem>
#include <fstream>
#include <iterator>
#include <unistd.h>

#include "common/tests/native_test.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/chart/chartswidget.h"
#include "tools/cabana/ui/layout_manager.h"

namespace {
struct TestDirectory {
  std::filesystem::path path;
  TestDirectory() {
    char name[] = "/tmp/cabana-layout-XXXXXX";
    auto *dir = mkdtemp(name);
    REQUIRE(dir != nullptr);
    path = dir;
  }
  ~TestDirectory() { std::filesystem::remove_all(path); }
};

std::string readFile(const std::filesystem::path &path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void testLayouts() {
  TestDirectory directory;
  DummyStream stream;
  can = &stream;
  REQUIRE(dbc()->open(SourceSet{0}, "test", "VERSION \"\"\nNS_ :\nBS_:\nBU_: TEST\nBO_ 123 TEST: 8 TEST\n SG_ VALUE : 0|8@1+ (1,0) [0|255] \"\" TEST\n"));

  cabana::Layout layout;
  layout.current_tab_index = 1;
  cabana::LayoutCurve scaled;
  scaled.name = "/test/value";
  scaled.color_hex = "#123456";
  scaled.scale = 2;
  scaled.offset = 10;
  scaled.visible = false;
  cabana::LayoutCurve derivative;
  derivative.name = "/test/value";
  derivative.color_hex = "#abcdef";
  derivative.derivative = true;
  cabana::LayoutCurve custom;
  custom.name = "Custom curve";
  custom.color_hex = "#009e73";
  custom.custom_python = cabana::CustomPythonSeries{"/missing/value", {"/missing/enabled"}, "factor = 3", "return value * factor"};
  cabana::LayoutCurve can_curve;
  can_curve.name = "VALUE";
  can_curve.can_id = "0:7B";
  can_curve.color_hex = "#0072b2";

  cabana::LayoutPane first;
  first.title = "Mixed overlay";
  first.series_type = 1;
  first.y_limits = std::make_pair(-5.0, 30.0);
  first.curves = {scaled, derivative, can_curve};
  cabana::LayoutPane second;
  second.title = "Custom transform";
  second.curves = {custom};
  cabana::LayoutPane third;
  third.title = "Other tab";
  third.series_type = 2;
  third.curves = {derivative};
  layout.tabs = {{"Lateral", {first, second}}, {"Longitudinal", {third}}};

  const auto source = directory.path / "source.json";
  const auto saved = directory.path / "saved.json";
  const auto saved_again = directory.path / "saved-again.json";
  REQUIRE(cabana::LayoutManager::saveLayout(layout, source));
  ChartsWidget charts;
  charts.loadLayoutFile(source);
  auto captured = charts.captureLayout();
  REQUIRE(captured.tabs.size() == 2);
  REQUIRE(captured.tabs[0].name == "Lateral");
  REQUIRE(captured.tabs[0].panes.size() == 2);
  REQUIRE(captured.tabs[0].panes[0].title == "Mixed overlay");
  REQUIRE(captured.tabs[0].panes[0].curves.size() == 3);
  REQUIRE(captured.current_tab_index == 1);
  REQUIRE(cabana::LayoutManager::saveLayout(captured, saved));
  REQUIRE(readFile(source) == readFile(saved));
  ChartsWidget reloaded;
  reloaded.loadLayoutFile(saved);
  REQUIRE(cabana::LayoutManager::saveLayout(reloaded.captureLayout(), saved_again));
  REQUIRE(readFile(saved) == readFile(saved_again));

  cabana::CerealSeriesMap samples;
  samples["/test/value"].samples = {{0, 1}, {1000000000, 3}, {2000000000, 6}};
  stream.cereal_series.replaceSegment(0, std::move(samples));
  ChartView values({0, 3}, &charts);
  values.addLayoutCurve(scaled);
  values.addLayoutCurve(derivative);
  REQUIRE(values.signals()[0].vals.size() == 3);
  REQUIRE(values.signals()[0].vals[0].y == 12);
  REQUIRE(values.signals()[0].vals[2].y == 22);
  REQUIRE(values.signals()[1].vals.size() == 2);
  REQUIRE(values.signals()[1].vals[0].y == 2);
  REQUIRE(values.signals()[1].vals[1].y == 3);
  REQUIRE(!cabana::LayoutManager::saveLayout(layout, directory.path));
}
}  // namespace

int main() {
  return run_native_test(testLayouts);
}
