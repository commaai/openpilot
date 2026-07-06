#include "tools/loggy/backend/route.h"
#include "tools/loggy/backend/store.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

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
    loggy::RouteIngestConfig config;
    config.selector = loggy::LogSelector::Auto;
    config.max_segments = 1;
    config.worker_count = 2;

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

    loggy::Store store;
    loggy::SegmentScheduler scheduler(&store);
    loggy::RouteIngestor ingestor(&scheduler);

    const auto start_ = std::chrono::steady_clock::now();
    ingestor.start(config);

    size_t timeline_span_count = 0;
    size_t log_count = 0;
    loggy::RouteIngestStatus status;
    // Poll the same public boundary the UI does: status() for lifecycle, drain_*() for data.
    while (true) {
      status = ingestor.status();
      timeline_span_count += ingestor.drain_timeline_spans().size();
      log_count += ingestor.drain_log_entries().size();
      store.begin_frame();
      if (status.state == loggy::RouteIngestState::Completed ||
          status.state == loggy::RouteIngestState::Failed ||
          status.state == loggy::RouteIngestState::Canceled) {
        break;
      }
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count() > 120.0) {
        std::cerr << "route ingest smoke timed out waiting for completion\n";
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ingestor.stop();

    const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    std::cout << "state=" << loggy::route_ingest_state_label(status.state)
              << " resolved_segments=" << status.segments_resolved
              << " loaded_segments=" << status.segments_loaded
              << " failed_segments=" << status.segments_failed
              << " first_segment_seconds=" << status.first_segment_seconds
              << " total_seconds=" << total_seconds
              << " timeline_spans=" << timeline_span_count
              << " logs=" << log_count
              << " store_series=" << store.series_path_count()
              << " store_can_ids=" << store.can_message_count()
              << "\n";

    if (status.state != loggy::RouteIngestState::Completed) {
      std::cerr << "route_ingest_smoke: " << (status.error.empty() ? "ingest did not complete" : status.error) << "\n";
      return 1;
    }
    if (status.segments_loaded == 0 || store.series_path_count() == 0) {
      std::cerr << "route ingest smoke produced no usable series\n";
      return 1;
    }
    if (timeline_span_count == 0) {
      std::cerr << "route ingest smoke produced no timeline spans\n";
      return 1;
    }
    if (log_count == 0) {
      std::cerr << "route ingest smoke produced no logs\n";
      return 1;
    }
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "route_ingest_smoke: " << err.what() << "\n";
    return 1;
  }
}
