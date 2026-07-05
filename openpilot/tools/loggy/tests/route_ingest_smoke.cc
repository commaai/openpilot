#include "tools/loggy/backend/route.h"
#include "tools/loggy/backend/store.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

constexpr const char *DEMO_ROUTE = "5beb9b58bd12b691/0000010a--a51155e496";

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [--demo] [--data-dir <dir>] [--selector <auto|rlog|qlog>]"
      << " [--max-segments <n>] [route]\n";
}

int parse_int(const char *value) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || end == nullptr || *end != '\0') throw std::runtime_error("invalid integer");
  return static_cast<int>(parsed);
}

loggy::LogSelector parse_selector(const std::string &value) {
  if (value == "auto") return loggy::LogSelector::Auto;
  if (value == "rlog") return loggy::LogSelector::RLog;
  if (value == "qlog") return loggy::LogSelector::QLog;
  throw std::runtime_error("invalid selector: " + value);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    loggy::RouteResolveConfig config;
    config.selector = loggy::LogSelector::Auto;
    config.max_segments = 1;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto require_value = [&](const char *flag) -> const char * {
        if (i + 1 >= argc) {
          usage(argv[0]);
          throw std::runtime_error(std::string("missing value for ") + flag);
        }
        return argv[++i];
      };

      if (arg == "--demo") {
        config.route_name = DEMO_ROUTE;
      } else if (arg == "--data-dir") {
        config.data_dir = require_value("--data-dir");
      } else if (arg == "--selector") {
        config.selector = parse_selector(require_value("--selector"));
      } else if (arg == "--max-segments") {
        config.max_segments = parse_int(require_value("--max-segments"));
      } else if (arg == "--help" || arg == "-h") {
        usage(argv[0]);
        return 0;
      } else if (!arg.empty() && arg[0] != '-' && config.route_name.empty()) {
        config.route_name = arg;
      } else {
        usage(argv[0]);
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (config.route_name.empty()) config.route_name = DEMO_ROUTE;

    const auto start = std::chrono::steady_clock::now();
    loggy::RouteResolveResult resolved = loggy::resolveRouteSegments(config);
    loggy::Store store;
    loggy::SegmentScheduler scheduler(&store);
    scheduler.setRouteSegments(resolved.segments);

    size_t loaded_segments = 0;
    double first_segment_seconds = 0.0;
    while (std::optional<loggy::SegmentWorkItem> work = scheduler.takeNext()) {
      loggy::SegmentLoadOptions options;
      options.extract.segment = work->segment;
      options.extract.coverage = work->range;
      loggy::SegmentLoadResult loaded = loggy::loadRouteSegment(*work, options);
      scheduler.publish(std::move(loaded.batch));
      store.beginFrame();
      ++loaded_segments;
      if (loaded_segments == 1) {
        first_segment_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      }
      std::cout << "segment " << work->segment
                << " events=" << loaded.event_count
                << " appended=" << loaded.appended_event_count
                << " series=" << loaded.series_count
                << " can_ids=" << loaded.can_message_count
                << " timeline_spans=" << loaded.timeline_span_count
                << " logs=" << loaded.log_count
                << " download=" << loaded.download_seconds
                << " parse=" << loaded.parse_seconds
                << " extract=" << loaded.extract_seconds
                << "\n";
      if (loaded.timeline_span_count == 0) {
        std::cerr << "route ingest smoke produced no timeline spans\n";
        return 1;
      }
      if (loaded.log_count == 0) {
        std::cerr << "route ingest smoke produced no logs\n";
        return 1;
      }
    }

    const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "resolved_segments=" << resolved.segments.size()
              << " loaded_segments=" << loaded_segments
              << " first_segment_seconds=" << first_segment_seconds
              << " total_seconds=" << total_seconds
              << " store_series=" << store.seriesPathCount()
              << " store_can_ids=" << store.canMessageCount()
              << "\n";

    if (loaded_segments == 0 || store.seriesPathCount() == 0) {
      std::cerr << "route ingest smoke produced no usable series\n";
      return 1;
    }
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "route_ingest_smoke: " << err.what() << "\n";
    return 1;
  }
}
