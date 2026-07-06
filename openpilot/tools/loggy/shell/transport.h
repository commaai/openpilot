#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace loggy {

inline constexpr double kDefaultPlaybackStepSeconds = 0.1;
inline constexpr double kMinPlaybackStepSeconds = 0.001;
inline constexpr double kMinViewSpanSeconds = 0.001;
inline constexpr double kPlaybackRatePresets[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0};

struct TimeRange {
  double start_ = 0.0;
  double end = 0.0;

  bool valid() const;
  double span() const;
};

TimeRange normalize_time_range(TimeRange range, double fallback_span = 1.0);
TimeRange clamp_time_range(TimeRange range, TimeRange bounds, double min_span = kMinViewSpanSeconds);
double clamp_time_to_range(double time, TimeRange range);
double fraction_to_time(double fraction, TimeRange range);
double time_to_fraction(double time, TimeRange range);

class PlaybackClock {
public:
  PlaybackClock();
  void init(TimeRange route_range);

  const TimeRange &route_range() const { return route_range_; }
  void set_route_range(TimeRange route_range);

  double tracker_time() const { return tracker_time_; }
  void seek(double time);

  bool playing() const { return playing_; }
  void play();
  void pause();
  bool toggle_playing();
  void set_playing(bool playing);

  double rate() const { return rate_; }
  void set_rate(double rate);

  bool loop() const { return loop_; }
  void set_loop(bool loop);
  bool has_loop_range() const { return loop_range_.has_value(); }
  TimeRange effective_loop_range() const;
  void set_loop_range(TimeRange loop_range);
  void clear_loop_range();

  double step_seconds() const { return step_seconds_; }
  void set_step_seconds(double seconds);
  double step_forward();
  double step_backward();
  double step(int direction);

  void advance(double delta_seconds);

private:
  TimeRange route_range_ = {0.0, 1.0};
  std::optional<TimeRange> loop_range_;
  double tracker_time_ = 0.0;
  double rate_ = 1.0;
  double step_seconds_ = kDefaultPlaybackStepSeconds;
  bool playing_ = false;
  bool loop_ = false;
};

class SharedViewRange {
public:
  SharedViewRange();
  explicit SharedViewRange(TimeRange route_range);

  const TimeRange &route_range() const { return route_range_; }
  const TimeRange &range() const { return range_; }

  double min_span() const { return min_span_; }
  void set_min_span(double min_span);
  void set_route_range(TimeRange route_range);
  void set_range(TimeRange range);
  void reset_to_route();
  bool contains(double time) const;

private:
  TimeRange route_range_ = {0.0, 1.0};
  TimeRange range_ = {0.0, 1.0};
  double min_span_ = kMinViewSpanSeconds;
};

enum class AlertLevel : uint8_t {
  Info,
  Warning,
  Critical,
};

enum class TimelineSpanKind : uint8_t {
  None,
  Engaged,
  AlertInfo,
  AlertWarning,
  AlertCritical,
};

struct EngagementSpan {
  double start_time = 0.0;
  double end_time = 0.0;
};

struct AlertSpan {
  double start_time = 0.0;
  double end_time = 0.0;
  AlertLevel level = AlertLevel::Info;
};

struct TimelineSpan {
  double start_time = 0.0;
  double end_time = 0.0;
  TimelineSpanKind kind = TimelineSpanKind::None;
};

struct TimelineColor {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 255;
};

struct TimelineRenderSpan {
  double start_fraction = 0.0;
  double end_fraction = 0.0;
  TimelineSpanKind kind = TimelineSpanKind::None;
  TimelineColor color;
};

TimelineSpanKind timeline_kind_for_alert(AlertLevel level);
TimelineColor timeline_span_color(TimelineSpanKind kind, uint8_t alpha = 255);
const char *timeline_span_label(TimelineSpanKind kind);

class TimelineModel {
public:
  TimelineModel();
  explicit TimelineModel(TimeRange route_range);

  const TimeRange &route_range() const { return route_range_; }
  void set_route_range(TimeRange route_range);

  const std::vector<TimelineSpan> &spans() const { return spans_; }
  void clear_spans();
  void set_spans(std::vector<TimelineSpan> spans);
  void set_spans(const std::vector<EngagementSpan> &engagements, const std::vector<AlertSpan> &alerts);

  double time_from_fraction(double fraction) const;
  double fraction_from_time(double time) const;
  double time_from_pixel(double pixel_x, double pixel_width) const;
  double pixel_from_time(double time, double pixel_width) const;
  TimelineSpanKind kind_at_time(double time) const;

  std::vector<TimelineRenderSpan> render_spans() const;
  std::vector<TimelineRenderSpan> render_spans(TimeRange display_range) const;

private:
  void clamp_stored_spans();

  TimeRange route_range_ = {0.0, 1.0};
  std::vector<TimelineSpan> spans_;
};

}  // namespace loggy
