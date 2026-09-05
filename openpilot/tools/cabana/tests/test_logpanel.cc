#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <unistd.h>

#include "imgui.h"
#include "implot.h"
#include "common/tests/native_test.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/chart/logpanel.h"

class TestLogStream : public DummyStream {
public:
  bool logSignalsEnabled() const override { return true; }
  double maxSeconds() const override { return 1; }
  void setData() {
    auto signals = std::make_shared<cabana::LogSignals>();
    (*signals)["carState/vEgo"] = {{0, 0}, {100000000, 1}, {200000000, std::numeric_limits<double>::quiet_NaN()}, {300000000, 3}};
    log_segments_ = {{0, signals}};
    ++log_revision_;
  }
  void evict() { log_segments_.clear(); ++log_revision_; }
};

void drawFrame(LogPanel &panel) {
  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 700));
  ImGui::Begin("Log Signals");
  panel.draw();
  ImGui::End();
  ImGui::Render();
  const auto *data = ImGui::GetDrawData();
  REQUIRE(data->TotalVtxCount > 0);
  for (const auto *list : data->CmdLists) {
    for (const auto &vertex : list->VtxBuffer) {
      REQUIRE(std::isfinite(vertex.pos.x) && std::isfinite(vertex.pos.y));
    }
  }
}

int main() {
  return run_native_test([] {
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1100, 800);
    io.DeltaTime = 1.0f / 60;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    io.Fonts->SetTexID(1);
    {
      TestLogStream stream;
      can = &stream;
      LogPanel panel;
      drawFrame(panel);
      stream.setData();
      char path[] = "/tmp/cabana-log-layout-XXXXXX";
      int fd = mkstemp(path);
      REQUIRE(fd >= 0);
      close(fd);
      std::ofstream(path) << R"({"version":1,"plots":[[{"signal":"carState/vEgo"},{"signal":"carState/vEgo","derivative":true}],[{"signal":"missing"}]]})";
      panel.loadLayout(path);
      std::remove(path);
      drawFrame(panel);
      drawFrame(panel);
      stream.evict();
      drawFrame(panel);
      can = nullptr;
    }
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    std::cout << "Log panel headless rendering passed (empty, non-finite, missing, evicted data)\n";
  });
}
