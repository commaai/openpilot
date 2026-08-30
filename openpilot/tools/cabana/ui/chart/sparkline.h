#pragma once

#include <vector>

#include "imgui.h"
#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/streams/abstractstream.h"

class Sparkline {
public:
  void update(const cabana::Signal *sig, CanEventIter first, CanEventIter last, int range, ImVec2 sz, double window_end);
  inline double freq() const { return freq_; }
  bool isEmpty() const { return render_points_.empty(); }
  // emits the rendered polyline at pos (top-left, screen coordinates)
  void draw(ImDrawList *draw_list, ImVec2 pos) const;

  ImVec2 size = {};  // empty when isEmpty()
  double min_val = 0;
  double max_val = 0;

private:
  struct Point {
    double x, y;
  };
  void render(const CabanaColor &color, int range, ImVec2 sz, double window_end);

  std::vector<Point> points_;
  std::vector<ImVec2> render_points_;
  ImU32 color_ = 0;
  double window_end_ = 0;  // the time the polyline was built for, so draw() can scroll it on from there
  double xscale_ = 0;
  bool draw_individual_points_ = false;
  double freq_ = 0;
};
