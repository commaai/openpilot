#include "tools/loggy/shell/transport.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool near(double a, double b, double eps = 1.0e-9) {
  return std::abs(a - b) <= eps;
}

void assert_near(double a, double b) {
  assert(near(a, b));
}

void test_play_advance_and_pause() {
  loggy::PlaybackClock clock;
  clock.init(loggy::TimeRange{0.0, 10.0});
  clock.seek(2.0);
  clock.play();

  clock.advance(1.5);
  assert_near(clock.tracker_time(), 3.5);
  assert(clock.playing());

  clock.pause();
  clock.advance(5.0);
  assert_near(clock.tracker_time(), 3.5);
}

void test_step_and_clamp() {
  loggy::PlaybackClock clock;
  clock.init(loggy::TimeRange{0.0, 2.0});
  clock.set_step_seconds(0.5);
  clock.seek(1.0);

  assert_near(clock.step_forward(), 1.5);
  assert_near(clock.step_backward(), 1.0);
  clock.seek(-10.0);
  assert_near(clock.tracker_time(), 0.0);
  assert_near(clock.step_backward(), 0.0);
  clock.seek(10.0);
  assert_near(clock.tracker_time(), 2.0);
  assert_near(clock.step_forward(), 2.0);
}

void test_boundary_and_looping() {
  loggy::PlaybackClock clock;
  clock.init(loggy::TimeRange{0.0, 10.0});
  clock.seek(9.5);
  clock.play();

  clock.advance(1.0);
  assert_near(clock.tracker_time(), 10.0);
  assert(!clock.playing());

  loggy::PlaybackClock loop;
  loop.init(loggy::TimeRange{0.0, 10.0});
  loop.set_loop(true);
  loop.set_loop_range(loggy::TimeRange{2.0, 4.0});
  loop.seek(3.75);
  loop.play();

  loop.advance(0.5);
  assert(loop.playing());
  assert_near(loop.tracker_time(), 2.0);
}

void test_view_range_independence() {
  loggy::PlaybackClock clock;
  clock.init(loggy::TimeRange{0.0, 20.0});
  loggy::SharedViewRange view(loggy::TimeRange{0.0, 20.0});
  view.set_range(loggy::TimeRange{5.0, 10.0});

  clock.seek(6.0);
  clock.play();
  clock.advance(3.0);

  assert_near(clock.tracker_time(), 9.0);
  assert_near(view.range().start_, 5.0);
  assert_near(view.range().end, 10.0);
}

void test_seek_mapping_and_timeline_spans() {
  loggy::TimelineModel timeline(loggy::TimeRange{100.0, 200.0});

  assert_near(timeline.time_from_fraction(0.25), 125.0);
  assert_near(timeline.time_from_fraction(-1.0), 100.0);
  assert_near(timeline.time_from_fraction(2.0), 200.0);
  assert_near(timeline.fraction_from_time(150.0), 0.5);
  assert_near(timeline.time_from_pixel(25.0, 100.0), 125.0);
  assert_near(timeline.pixel_from_time(175.0, 200.0), 150.0);

  const std::vector<loggy::EngagementSpan> engagements = {{110.0, 130.0}};
  const std::vector<loggy::AlertSpan> alerts = {{120.0, 140.0, loggy::AlertLevel::Critical}};
  timeline.set_spans(engagements, alerts);

  assert(timeline.spans().size() == 2);
  assert(timeline.kind_at_time(115.0) == loggy::TimelineSpanKind::Engaged);
  assert(timeline.kind_at_time(125.0) == loggy::TimelineSpanKind::AlertCritical);
  assert(timeline.kind_at_time(150.0) == loggy::TimelineSpanKind::None);

  const std::vector<loggy::TimelineRenderSpan> full = timeline.render_spans();
  assert(full.size() == 2);
  assert_near(full[0].start_fraction, 0.10);
  assert_near(full[0].end_fraction, 0.30);
  assert(full[0].kind == loggy::TimelineSpanKind::Engaged);
  assert_near(full[1].start_fraction, 0.20);
  assert_near(full[1].end_fraction, 0.40);
  assert(full[1].kind == loggy::TimelineSpanKind::AlertCritical);

  const std::vector<loggy::TimelineRenderSpan> visible = timeline.render_spans(loggy::TimeRange{120.0, 160.0});
  assert(visible.size() == 2);
  assert_near(visible[0].start_fraction, 0.0);
  assert_near(visible[0].end_fraction, 0.25);
  assert_near(visible[1].start_fraction, 0.0);
  assert_near(visible[1].end_fraction, 0.5);

  const loggy::TimelineColor critical = loggy::timeline_span_color(loggy::TimelineSpanKind::AlertCritical);
  assert(critical.r == 199 && critical.g == 0 && critical.b == 57 && critical.a == 255);
}

}  // namespace

int main() {
  test_play_advance_and_pause();
  test_step_and_clamp();
  test_boundary_and_looping();
  test_view_range_independence();
  test_seek_mapping_and_timeline_spans();
  std::cout << "transport_smoke passed\n";
  return 0;
}
