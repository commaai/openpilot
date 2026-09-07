#include "common/tests/native_test.h"
#include "tools/cabana/ui/widgets/cameraview.h"

void test_video_edges() {
  ImDrawListSharedData shared;
  shared.InitialFlags = ImDrawListFlags_AntiAliasedFill;
  shared.SetCircleTessellationMaxError(0.3f);
  ImDrawList draw(&shared);
  const ImTextureRef texture((ImTextureID)1);
  for (bool crop : {false, true}) {
    for (bool mirror : {false, true}) {
      auto placement = videoPlacement(ImRect(10, 20, 422, 252), DEFAULT_CAMERA_ASPECT_RATIO, crop);
      if (mirror) std::swap(placement.uv0.x, placement.uv1.x);
      const float radius = 6;
      auto straight = [&](const ImDrawVert &v) {
        return (v.pos.x >= placement.min.x + radius - 1 && v.pos.x <= placement.max.x - radius + 1) ||
               (v.pos.y >= placement.min.y + radius - 1 && v.pos.y <= placement.max.y - radius + 1);
      };
      draw._ResetForNewFrame();
      draw.Flags = ImDrawListFlags_AntiAliasedFill;
      draw.PushClipRectFullScreen();
      draw.AddImageRounded(texture, placement.min, placement.max, placement.uv0, placement.uv1, IM_COL32_WHITE, radius);
      bool original_fringe = false;
      for (const auto &v : draw.VtxBuffer) original_fringe |= straight(v) && (v.col & IM_COL32_A_MASK) == 0;
      REQUIRE(original_fringe);

      draw._ResetForNewFrame();
      draw.Flags = ImDrawListFlags_AntiAliasedFill;
      draw.PushClipRectFullScreen();
      drawVideoImage(&draw, texture, placement, radius);
      bool corner_fringe = false;
      for (const auto &v : draw.VtxBuffer) {
        if (straight(v)) {
          REQUIRE((v.col & IM_COL32_A_MASK) == IM_COL32_A_MASK);
          REQUIRE(v.pos.x >= placement.min.x && v.pos.x <= placement.max.x);
          REQUIRE(v.pos.y >= placement.min.y && v.pos.y <= placement.max.y);
        } else {
          corner_fringe |= (v.col & IM_COL32_A_MASK) == 0;
        }
        REQUIRE(v.uv.x >= 0 && v.uv.x <= 1);
        REQUIRE(v.uv.y >= 0 && v.uv.y <= 1);
      }
      REQUIRE(corner_fringe);
    }
  }
}

int main() { return run_native_test(test_video_edges); }
