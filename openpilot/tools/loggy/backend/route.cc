#include "tools/loggy/backend/route.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <utility>

#include "json11/json11.hpp"
#include "kj/exception.h"
#include "tools/replay/logreader.h"
#include "tools/replay/py_downloader.h"

namespace loggy {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct SegmentLogFiles {
  std::string rlog;
  std::string qlog;
  std::string road_cam;
  std::string driver_cam;
  std::string wide_road_cam;
  std::string qcamera;
};

constexpr double kNominalSegmentSeconds = 60.0;
constexpr const char *kCommaApiHost = "https://api.commadotai.com";

const std::array<RouteBrowserPeriod, 5> kRouteBrowserPeriods = {{
  {"Last week", 7},
  {"Last 2 weeks", 14},
  {"Last month", 30},
  {"Last 6 months", 180},
  {"Preserved", -1},
}};

std::string trim(std::string value) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), is_space));
  value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
  return value;
}

std::string formatEpochTime(double epoch_seconds, const char *format) {
  std::time_t epoch = static_cast<std::time_t>(epoch_seconds);
  std::tm tm {};
#if defined(_WIN32)
  gmtime_s(&tm, &epoch);
#else
  gmtime_r(&epoch, &tm);
#endif
  char buffer[64] = {};
  std::strftime(buffer, sizeof(buffer), format, &tm);
  return buffer;
}

double parseIso8601Epoch(const std::string &value) {
  std::tm tm {};
  std::istringstream stream(value);
  stream >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
  if (stream.fail()) return 0.0;
#if defined(_WIN32)
  return static_cast<double>(_mkgmtime(&tm));
#else
  return static_cast<double>(timegm(&tm));
#endif
}

bool parseSegmentNumber(std::string_view value, int &out) {
  if (value.empty()) return false;
  char *end = nullptr;
  const long parsed = std::strtol(std::string(value).c_str(), &end, 10);
  if (end == nullptr || *end != '\0') return false;
  out = static_cast<int>(parsed);
  return true;
}

bool parseNonnegativeSegmentNumber(std::string_view value, int &out) {
  int parsed = 0;
  if (!parseSegmentNumber(value, parsed) || parsed < 0) return false;
  out = parsed;
  return true;
}

bool isLogSelectorChar(char c) {
  return c == 'a' || c == 'r' || c == 'q';
}

LogSelector parseLogSelectorChar(char c) {
  switch (c) {
    case 'r': return LogSelector::RLog;
    case 'q': return LogSelector::QLog;
    case 'a':
    default: return LogSelector::Auto;
  }
}

std::string fileBaseName(const std::string &file) {
  const size_t query_pos = file.find('?');
  const std::string path = query_pos == std::string::npos ? file : file.substr(0, query_pos);
  const size_t slash_pos = path.find_last_of("/\\");
  std::string name = slash_pos == std::string::npos ? path : path.substr(slash_pos + 1);
  const size_t marker = name.rfind("--");
  if (marker != std::string::npos) name = name.substr(marker + 2);
  return name;
}

void addLogFileToSegments(std::map<int, SegmentLogFiles> *segments, int segment_number, const std::string &file) {
  const std::string name = fileBaseName(file);
  SegmentLogFiles &segment = (*segments)[segment_number];
  if (name == "rlog.bz2" || name == "rlog.zst" || name == "rlog") {
    segment.rlog = file;
  } else if (name == "qlog.bz2" || name == "qlog.zst" || name == "qlog") {
    segment.qlog = file;
  } else if (name == "fcamera.hevc") {
    segment.road_cam = file;
  } else if (name == "dcamera.hevc") {
    segment.driver_cam = file;
  } else if (name == "ecamera.hevc") {
    segment.wide_road_cam = file;
  } else if (name == "qcamera.ts") {
    segment.qcamera = file;
  }
}

std::map<int, SegmentLogFiles> trimSegments(std::map<int, SegmentLogFiles> segments, const RouteSelection &route) {
  if (route.begin_segment > 0) {
    segments.erase(segments.begin(), segments.lower_bound(route.begin_segment));
  }
  if (route.end_segment >= 0) {
    segments.erase(segments.upper_bound(route.end_segment), segments.end());
  }
  return segments;
}

std::map<int, SegmentLogFiles> loadSegmentsFromJson(const json11::Json &json) {
  std::map<int, SegmentLogFiles> segments;
  static const std::regex rx(R"(\/(\d+)\/)");
  for (const auto &value : json.object_items()) {
    for (const auto &url : value.second.array_items()) {
      const std::string url_str = url.string_value();
      std::smatch match;
      if (!std::regex_search(url_str, match, rx)) continue;
      addLogFileToSegments(&segments, std::stoi(match[1].str()), url_str);
    }
  }
  return segments;
}

std::map<int, SegmentLogFiles> loadSegmentsFromServer(const RouteSelection &route) {
  const std::string result = PyDownloader::getRouteFiles(route.canonical_name);
  if (result.empty()) throw std::runtime_error("Failed to fetch route files for " + route.canonical_name);

  std::string parse_error;
  const json11::Json json = json11::Json::parse(result, parse_error);
  if (!parse_error.empty()) throw std::runtime_error("Failed to parse route file list for " + route.canonical_name);
  if (json.is_object() && json["error"].is_string()) {
    throw std::runtime_error("Route API error for " + route.canonical_name + ": " + json["error"].string_value());
  }
  return loadSegmentsFromJson(json);
}

std::map<int, SegmentLogFiles> loadSegmentsFromLocal(const RouteSelection &route, const std::string &data_dir) {
  std::map<int, SegmentLogFiles> segments;
  const std::string pattern = route.timestamp + "--";
  for (const auto &entry : fs::directory_iterator(data_dir)) {
    if (!entry.is_directory()) continue;
    const std::string dirname = entry.path().filename().string();
    if (dirname.find(pattern) == std::string::npos) continue;
    const size_t marker = dirname.rfind("--");
    if (marker == std::string::npos) continue;
    int segment_number = 0;
    if (!parseSegmentNumber(dirname.substr(marker + 2), segment_number)) continue;
    for (const auto &file : fs::directory_iterator(entry.path())) {
      if (file.is_regular_file()) addLogFileToSegments(&segments, segment_number, file.path().string());
    }
  }
  return segments;
}

const std::string &selectedLogPath(const SegmentLogFiles &segment, LogSelector selector) {
  switch (selector) {
    case LogSelector::RLog:
      return segment.rlog;
    case LogSelector::QLog:
      return segment.qlog;
    case LogSelector::Auto:
    default:
      return !segment.rlog.empty() ? segment.rlog : segment.qlog;
  }
}

TimeRange routeRangeForSegmentCount(size_t count) {
  if (count == 0) return {};
  return {0.0, static_cast<double>(count) * kNominalSegmentSeconds};
}

double secondsSince(Clock::time_point start_) {
  return std::chrono::duration<double>(Clock::now() - start_).count();
}

size_t boundedWorkerCount(size_t requested, size_t segment_count) {
  if (segment_count == 0) return 0;
  if (requested == 0) requested = 1;
  return std::min(requested, segment_count);
}

void mark_loaded(RouteIngestStatus *status) {
  ++status->segments_loaded;
}

void mark_failed(RouteIngestStatus *status) {
  ++status->segments_failed;
}

void markPublished(RouteIngestStatus *status) {
  ++status->batches_published;
}

uint8_t operatingSystemPriorityToLevel(uint8_t priority) {
  switch (priority) {
    case 0:
    case 1:
    case 2:
      return 50;
    case 3:
      return 40;
    case 4:
      return 20;
    case 5:
      return 30;
    case 6:
      return 40;
    case 7:
    default:
      return 50;
  }
}

uint8_t alertStatusToLevel(cereal::SelfdriveState::AlertStatus status) {
  switch (status) {
    case cereal::SelfdriveState::AlertStatus::NORMAL:
      return 20;
    case cereal::SelfdriveState::AlertStatus::USER_PROMPT:
      return 30;
    case cereal::SelfdriveState::AlertStatus::CRITICAL:
      return 40;
  }
  return 20;
}

double operatingSystemWallTimeSeconds(uint64_t timestamp) {
  if (timestamp == 0) return 0.0;
  if (timestamp > 1000000000000ULL) return static_cast<double>(timestamp) / 1.0e9;
  if (timestamp > 1000000000ULL) return static_cast<double>(timestamp) / 1.0e6;
  return static_cast<double>(timestamp);
}

std::string alertMessageText(const cereal::SelfdriveState::Reader &state) {
  std::string text = state.getAlertText1().cStr();
  const std::string text2 = state.getAlertText2().cStr();
  if (!text2.empty()) {
    if (!text.empty()) text += " - ";
    text += text2;
  }
  return text;
}

LogEntry makeLogEntry(const cereal::Event::Reader &event, double time_offset, LogOrigin origin, uint8_t level = 20) {
  const double boot_time = static_cast<double>(event.getLogMonoTime()) / 1.0e9;
  LogEntry entry;
  entry.mono_time = boot_time - time_offset;
  entry.boot_time = boot_time;
  entry.origin = origin;
  entry.level = level;
  return entry;
}

void appendLogEvent(cereal::Event::Which which,
                    const cereal::Event::Reader &event,
                    double time_offset,
                    std::vector<LogEntry> *logs,
                    std::string &last_alert_key) {
  if (logs == nullptr) return;

  switch (which) {
    case cereal::Event::Which::LOG_MESSAGE:
    case cereal::Event::Which::ERROR_LOG_MESSAGE: {
      const std::string raw = which == cereal::Event::Which::LOG_MESSAGE
        ? event.getLogMessage().cStr()
        : event.getErrorLogMessage().cStr();
      LogEntry entry = makeLogEntry(event, time_offset, LogOrigin::Log,
                                    which == cereal::Event::Which::ERROR_LOG_MESSAGE ? 40 : 20);
      entry.source = "log";
      entry.message = raw;
      std::string err;
      const json11::Json parsed = json11::Json::parse(raw, err);
      if (err.empty() && parsed.is_object()) {
        if (parsed["created"].is_number()) entry.wall_time = parsed["created"].number_value();
        if (parsed["levelnum"].is_number()) entry.level = static_cast<uint8_t>(parsed["levelnum"].int_value());
        const std::string filename = parsed["filename"].string_value();
        const int line = parsed["lineno"].is_number() ? parsed["lineno"].int_value() : 0;
        entry.source = filename.empty() ? "log" : filename + (line > 0 ? ":" + std::to_string(line) : "");
        entry.func = parsed["funcname"].string_value();
        if (parsed["msg"].is_string()) entry.message = parsed["msg"].string_value();
        if (!parsed["ctx"].is_null()) entry.context = parsed["ctx"].dump();
      }
      logs->push_back(std::move(entry));
      break;
    }
    case cereal::Event::Which::OPERATING_SYSTEM_LOG: {
      const auto os_log = event.getOperatingSystemLog();
      LogEntry entry = makeLogEntry(event, time_offset, LogOrigin::OperatingSystem,
                                    operatingSystemPriorityToLevel(os_log.getPriority()));
      entry.wall_time = operatingSystemWallTimeSeconds(os_log.getTs());
      entry.source = os_log.hasTag() ? os_log.getTag().cStr() : "operating_system";
      entry.message = os_log.hasMessage() ? os_log.getMessage().cStr() : std::string();
      entry.context = "pid=" + std::to_string(os_log.getPid()) + ", tid=" + std::to_string(os_log.getTid());

      std::string err;
      const json11::Json parsed = json11::Json::parse(entry.message, err);
      if (err.empty() && parsed.is_object()) {
        if (parsed["MESSAGE"].is_string()) entry.message = parsed["MESSAGE"].string_value();
        if (parsed["SYSLOG_IDENTIFIER"].is_string() && !parsed["SYSLOG_IDENTIFIER"].string_value().empty()) {
          entry.source = parsed["SYSLOG_IDENTIFIER"].string_value();
        }
        if (parsed["PRIORITY"].is_number()) {
          entry.level = operatingSystemPriorityToLevel(static_cast<uint8_t>(parsed["PRIORITY"].int_value()));
        }
      }
      logs->push_back(std::move(entry));
      break;
    }
    case cereal::Event::Which::SELFDRIVE_STATE: {
      const auto sd = event.getSelfdriveState();
      const std::string alert_type = sd.getAlertType().cStr();
      const std::string alert_text1 = sd.getAlertText1().cStr();
      const std::string alert_text2 = sd.getAlertText2().cStr();
      if (alert_text1.empty() && alert_type.empty()) break;
      const std::string key = alert_type + "\n" + alert_text1 + "\n" + alert_text2;
      if (key == last_alert_key) break;
      last_alert_key = key;
      LogEntry entry = makeLogEntry(event, time_offset, LogOrigin::Alert, alertStatusToLevel(sd.getAlertStatus()));
      entry.source = "alert";
      entry.func = alert_type;
      entry.message = alertMessageText(sd);
      logs->push_back(std::move(entry));
      break;
    }
    default:
      break;
  }
}

TimelineSpanKind timelineKindForSelfdrive(cereal::SelfdriveState::AlertStatus status, bool enabled) {
  if (!enabled) return TimelineSpanKind::None;
  switch (status) {
    case cereal::SelfdriveState::AlertStatus::NORMAL:
      return TimelineSpanKind::Engaged;
    case cereal::SelfdriveState::AlertStatus::USER_PROMPT:
      return TimelineSpanKind::AlertInfo;
    case cereal::SelfdriveState::AlertStatus::CRITICAL:
      return TimelineSpanKind::AlertCritical;
  }
  return TimelineSpanKind::Engaged;
}

void appendTimelinePoint(std::vector<TimelineSpan> *spans, double mono_time, TimelineSpanKind kind) {
  if (spans == nullptr) return;
  if (kind == TimelineSpanKind::None) return;
  if (!spans->empty() && spans->back().kind == kind) {
    spans->back().end_time = std::max(spans->back().end_time, mono_time);
    return;
  }
  spans->push_back({mono_time, mono_time, kind});
}

std::vector<TimelineSpan> extractTimelineSpans(const std::vector<Event> &events, double time_offset) {
  std::vector<TimelineSpan> spans;
  spans.reserve(events.size() / 32);
  for (const Event &event_record : events) {
    if (event_record.which != cereal::Event::Which::SELFDRIVE_STATE || event_record.eidx_segnum != -1) continue;
    try {
      capnp::FlatArrayMessageReader event_reader(event_record.data);
      const cereal::Event::Reader event = event_reader.getRoot<cereal::Event>();
      const auto selfdrive = event.getSelfdriveState();
      const double mono_time = static_cast<double>(event.getLogMonoTime()) / 1.0e9 - time_offset;
      appendTimelinePoint(&spans, mono_time, timelineKindForSelfdrive(selfdrive.getAlertStatus(), selfdrive.getEnabled()));
    } catch (const kj::Exception &) {
      continue;
    }
  }
  return spans;
}

std::vector<LogEntry> extractLogEntries(const std::vector<Event> &events, double time_offset) {
  std::vector<LogEntry> logs;
  logs.reserve(events.size() / 16);
  std::string last_alert_key;
  for (const Event &event_record : events) {
    if (event_record.eidx_segnum != -1) continue;
    try {
      capnp::FlatArrayMessageReader event_reader(event_record.data);
      const cereal::Event::Reader event = event_reader.getRoot<cereal::Event>();
      appendLogEvent(event_record.which, event, time_offset, &logs, last_alert_key);
    } catch (const kj::Exception &) {
      continue;
    }
  }
  std::sort(logs.begin(), logs.end(), [](const LogEntry &a, const LogEntry &b) {
    if (std::abs(a.mono_time - b.mono_time) > 1.0e-9) return a.mono_time < b.mono_time;
    return a.message < b.message;
  });
  return logs;
}

std::string extractCarFingerprint(const std::vector<::Event> &events) {
  for (const ::Event &event_record : events) {
    if (event_record.which != cereal::Event::Which::CAR_PARAMS || event_record.eidx_segnum != -1) continue;
    try {
      capnp::FlatArrayMessageReader event_reader(event_record.data);
      const cereal::Event::Reader event = event_reader.getRoot<cereal::Event>();
      const std::string fingerprint = event.getCarParams().getCarFingerprint().cStr();
      if (!fingerprint.empty()) return fingerprint;
    } catch (const kj::Exception &) {
      continue;
    }
  }
  return {};
}

}  // namespace

RouteSelection parse_route_selection(std::string route_name) {
  RouteSelection route;
  route_name = trim(std::move(route_name));
  if (route_name.size() >= 2 && route_name[route_name.size() - 2] == '/'
      && isLogSelectorChar(static_cast<char>(std::tolower(route_name.back())))) {
    route.selector = parseLogSelectorChar(static_cast<char>(std::tolower(route_name.back())));
    route.selector_explicit = true;
    route_name.resize(route_name.size() - 2);
  }

  static const std::regex pattern(R"(^(([a-z0-9]{16})[|_/])?(.{20})((--|/)((-?\d+(:(-?\d+)?)?)|(:-?\d+)))?$)");
  std::smatch match;
  if (!std::regex_match(route_name, match, pattern)) return route;

  route.dongle_id = match[2].str();
  route.timestamp = match[3].str();
  route.canonical_name = route.dongle_id + "|" + route.timestamp;

  const std::string separator = match[5].str();
  const std::string range_str = match[6].str();
  if (!range_str.empty()) {
    route.slice_explicit = true;
    if (separator == "/") {
      const size_t pos = range_str.find(':');
      int begin_segment = 0;
      if (!parseSegmentNumber(range_str.substr(0, pos), begin_segment)) return {};
      route.begin_segment = begin_segment;
      route.end_segment = begin_segment;
      if (pos != std::string::npos) {
        int end_segment = -1;
        const std::string end_str = range_str.substr(pos + 1);
        if (!end_str.empty() && !parseSegmentNumber(end_str, end_segment)) return {};
        route.end_segment = end_str.empty() ? -1 : end_segment;
      }
    } else if (separator == "--") {
      int begin_segment = 0;
      if (!parseSegmentNumber(range_str, begin_segment)) return {};
      route.begin_segment = begin_segment;
    }
  }
  return route;
}

std::optional<RouteSelection> route_selection_from_text(std::string_view route_name) {
  const RouteSelection parsed = parse_route_selection(std::string(route_name));
  if (parsed.timestamp.empty()) return std::nullopt;
  if (parsed.slice_explicit && (parsed.begin_segment < 0 || parsed.end_segment < -1 ||
                                (parsed.end_segment >= 0 && parsed.end_segment < parsed.begin_segment))) {
    return std::nullopt;
  }
  return parsed;
}

const std::array<RouteBrowserPeriod, 5> &route_browser_periods() {
  return kRouteBrowserPeriods;
}

std::string route_browser_device_routes_url(const std::string &dongle_id, int64_t start_ms, int64_t end_ms, bool preserved) {
  if (dongle_id.empty()) return {};
  const std::string base = std::string(kCommaApiHost) + "/v1/devices/" + dongle_id;
  if (preserved) return base + "/routes_/preserved";
  return base + "/routes_segments?start_=" + std::to_string(start_ms) + "&end=" + std::to_string(end_ms);
}

std::string route_browser_route_files_url(const std::string &route_name) {
  if (route_name.empty()) return {};
  return std::string(kCommaApiHost) + "/v1/route/" + route_name + "/files";
}

std::string route_browser_route_label(double from_epoch_sec, double to_epoch_sec) {
  if (from_epoch_sec <= 0.0 || to_epoch_sec <= 0.0) return {};
  const int minutes = std::max(0.0, to_epoch_sec - from_epoch_sec) / 60.0;
  const std::string date = formatEpochTime(from_epoch_sec, "%a %b %e %H:%M:%S %Y");
  char out[128];
  std::snprintf(out, sizeof(out), "%s    %dmin", date.c_str(), minutes);
  return out;
}

RouteBrowserParseResult parse_route_browser_routes(const std::string &json_text, bool preserved) {
  RouteBrowserParseResult result;
  std::string parse_error;
  const json11::Json doc = json11::Json::parse(json_text, parse_error);
  if (!parse_error.empty()) {
    result.second = "Network error";
    return result;
  }
  if (doc.is_object() && doc["error"].is_string()) {
    const std::string code = doc["error"].string_value();
    result.second = code == "unauthorized" ? "Unauthorized. Authenticate with openpilot/tools/lib/auth.py" : "Network error";
    return result;
  }
  if (!doc.is_array()) {
    result.second = "Network error";
    return result;
  }

  result.first.reserve(doc.array_items().size());
  for (const json11::Json &route : doc.array_items()) {
    const std::string fullname = route["fullname"].string_value();
    if (fullname.empty()) continue;
    const double from = preserved ? parseIso8601Epoch(route["start_time"].string_value())
                                 : route["start_time_utc_millis"].number_value() / 1000.0;
    const double to = preserved ? parseIso8601Epoch(route["end_time"].string_value())
                               : route["end_time_utc_millis"].number_value() / 1000.0;
    const std::string label = route_browser_route_label(from, to);
    result.first.push_back({label, fullname});
  }
  return result;
}

const char *log_selector_name(LogSelector selector) {
  switch (selector) {
    case LogSelector::RLog: return "r";
    case LogSelector::QLog: return "q";
    case LogSelector::Auto:
    default: return "a";
  }
}

const char *log_selector_description(LogSelector selector) {
  switch (selector) {
    case LogSelector::RLog: return "rlog only";
    case LogSelector::QLog: return "qlog only";
    case LogSelector::Auto:
    default: return "any rlog/qlog";
  }
}

char log_selector_char(LogSelector selector) {
  return log_selector_name(selector)[0];
}

std::string route_selection_display_slice(const RouteSelection &selection) {
  const int begin = selection.begin_segment;
  const int end = selection.end_segment;
  if (end < 0) return std::to_string(begin) + ":";
  if (end == begin) return std::to_string(begin);
  return std::to_string(begin) + ":" + std::to_string(end);
}

std::string route_selection_full_spec(const RouteSelection &selection) {
  if (selection.timestamp.empty()) return {};
  std::string spec = selection.dongle_id.empty()
    ? selection.timestamp
    : selection.dongle_id + "/" + selection.timestamp;
  if (selection.slice_explicit) {
    spec += "/";
    spec += route_selection_display_slice(selection);
  }
  if (selection.selector_explicit) {
    spec += "/";
    spec.push_back(log_selector_char(selection.selector));
  }
  return spec;
}

std::string route_useradmin_url(const RouteSelection &selection) {
  if (selection.dongle_id.empty() || selection.timestamp.empty()) return {};
  return "https://useradmin.comma.ai/?onebox=" + selection.dongle_id + "%7C" + selection.timestamp;
}

std::string route_connect_url(const RouteSelection &selection) {
  if (selection.dongle_id.empty() || selection.timestamp.empty()) return {};
  return "https://connect.comma.ai/" + selection.dongle_id + "/" + selection.timestamp;
}

bool parse_route_slice_spec_impl(std::string_view text, int &begin, int &end) {
  const std::string trimmed = trim(std::string(text));
  if (trimmed.empty()) return false;
  const size_t colon = trimmed.find(':');
  int parsed_begin = 0;
  if (!parseNonnegativeSegmentNumber(trimmed.substr(0, colon), parsed_begin)) return false;
  int parsed_end = parsed_begin;
  if (colon != std::string::npos) {
    const std::string end_text = trimmed.substr(colon + 1);
    if (end_text.empty()) {
      parsed_end = -1;
    } else if (!parseNonnegativeSegmentNumber(end_text, parsed_end) || parsed_end < parsed_begin) {
      return false;
    }
  }
  begin = parsed_begin;
  end = parsed_end;
  return true;
}

std::optional<RouteSliceSpec> parse_route_slice_spec(std::string_view text) {
  int parsed_begin = 0;
  int parsed_end = 0;
  if (!parse_route_slice_spec_impl(text, parsed_begin, parsed_end)) return std::nullopt;
  return RouteSliceSpec{parsed_begin, parsed_end};
}

RouteResolveResult resolve_route_segments(const RouteResolveConfig &config) {
  if (config.route_name.empty()) throw std::runtime_error("No route name provided");

  RouteSelection selection = parse_route_selection(config.route_name);
  if (selection.canonical_name.empty() || (config.data_dir.empty() && selection.dongle_id.empty())) {
    throw std::runtime_error("Invalid route format: " + config.route_name);
  }
  if (config.selector != LogSelector::Auto && !selection.selector_explicit) {
    selection.selector = config.selector;
  }

  std::map<int, SegmentLogFiles> logs = config.data_dir.empty()
    ? loadSegmentsFromServer(selection)
    : loadSegmentsFromLocal(selection, config.data_dir);
  logs = trimSegments(std::move(logs), selection);
  if (logs.empty()) throw std::runtime_error("No log segments found for " + config.route_name);

  RouteResolveResult result;
  result.selection = selection;
  result.segments.reserve(logs.size());
  size_t relative_index = 0;
  for (const auto &[segment_number, segment] : logs) {
    // Build synthetic timeline ranges from relative_index to keep ranges contiguous after trimming/slicing.
    if (config.max_segments >= 0 && static_cast<int>(result.segments.size()) >= config.max_segments) break;
    const std::string &path = selectedLogPath(segment, selection.selector);
    if (path.empty()) continue;
    result.segments.push_back(RouteSegment{
      .segment = segment_number,
      .range = {static_cast<double>(relative_index) * kNominalSegmentSeconds,
                static_cast<double>(relative_index + 1) * kNominalSegmentSeconds},
      .log_path = path,
      .cache_path = {},
      .road_camera_path = segment.road_cam,
      .driver_camera_path = segment.driver_cam,
      .wide_road_camera_path = segment.wide_road_cam,
      .qroad_camera_path = segment.qcamera,
      .state = SegmentState::Pending,
    });
    ++relative_index;
  }
  if (result.segments.empty()) throw std::runtime_error("No selected log files found for " + config.route_name);
  result.route_range = routeRangeForSegmentCount(result.segments.size());
  return result;
}

SegmentLoadResult load_route_segment(const SegmentWorkItem &work,
                                   const SegmentLoadOptions &options,
                                   std::atomic<bool> *abort) {
  if (work.log_path.empty()) {
    throw std::runtime_error("Missing log path for segment " + std::to_string(work.segment));
  }

  LogReader reader;
  // Pass abort flag into reader so expensive IO can stop promptly when the ingest is canceled.
  if (!reader.load(work.log_path, abort, options.local_cache)) {
    throw std::runtime_error("Failed to load log segment: " + work.log_path);
  }

  SegmentExtractOptions extract = options.extract;
  extract.segment = work.segment;
  extract.coverage = work.range;
  if (!extract.time_offset.has_value() && !reader.events.empty()) {
    extract.time_offset = static_cast<double>(reader.events.front().mono_time) / 1.0e9 - work.range.start_;
  }

  const auto extract_start = Clock::now();
  SegmentExtractResult extracted = extract_segment_series(reader.events, extract);

  SegmentLoadResult result;
  result.batch = std::move(extracted.batch);
  result.timeline_spans = extractTimelineSpans(reader.events, extract.time_offset_seconds());
  result.logs = extractLogEntries(reader.events, extract.time_offset_seconds());
  result.car_fingerprint = extractCarFingerprint(reader.events);
  result.event_count = extracted.events_seen;
  result.appended_event_count = extracted.events_appended;
  result.series_count = result.batch.series.size();
  result.can_message_count = result.batch.can_events.size();
  result.timeline_span_count = result.timeline_spans.size();
  result.log_count = result.logs.size();
  result.compressed_bytes = reader.compressed_size();
  result.decompressed_bytes = reader.decompressed_size();
  result.download_seconds = reader.download_seconds();
  result.decompress_seconds = reader.decompress_seconds();
  result.parse_seconds = reader.parse_seconds();
  result.extract_seconds = secondsSince(extract_start);
  return result;
}

const char *route_ingest_state_label(RouteIngestState state) {
  switch (state) {
    case RouteIngestState::Idle: return "idle";
    case RouteIngestState::Resolving: return "resolving";
    case RouteIngestState::Loading: return "loading";
    case RouteIngestState::Completed: return "complete";
    case RouteIngestState::Failed: return "failed";
    case RouteIngestState::Canceled: return "canceled";
  }
  return "unknown";
}

RouteIngestor::RouteIngestor(SegmentScheduler *scheduler) : scheduler_(scheduler) {}

RouteIngestor::~RouteIngestor() {
  stop();
}

void RouteIngestor::set_scheduler(SegmentScheduler *scheduler) {
  scheduler_ = scheduler;
}

void RouteIngestor::start(RouteIngestConfig config) {
  stop();
  // Cancel any in-flight worker set before starting a new run to avoid overlapping lifetime.
  abort_ = false;
  RouteIngestStatus next;
  next.state = RouteIngestState::Resolving;
  next.route_name = config.resolve.route_name;
  next.selection = parse_route_selection(config.resolve.route_name);
  if (config.resolve.selector != LogSelector::Auto && !next.selection.selector_explicit) {
    next.selection.selector = config.resolve.selector;
  }
  update_status(next);
  thread_ = std::thread(&RouteIngestor::run, this, std::move(config));
}

void RouteIngestor::stop() {
  // abort_ is atomic so workers can observe cancellation without extra lock traffic.
  abort_ = true;
  if (thread_.joinable()) thread_.join();
}

RouteIngestStatus RouteIngestor::status() const {
  // Lock status_ for a coherent snapshot across concurrent worker updates.
  std::lock_guard lock(status_mutex_);
  return status_;
}

std::vector<TimelineSpan> RouteIngestor::drain_timeline_spans() {
  // Drain by swap so queued spans remain valid after lock release and are not interleaved by producers.
  std::lock_guard lock(timeline_mutex_);
  std::vector<TimelineSpan> spans;
  spans.swap(staged_timeline_spans_);
  return spans;
}

std::vector<LogEntry> RouteIngestor::drain_log_entries() {
  // Drain by swap so queued logs remain valid after lock release and producers can continue appending.
  std::lock_guard lock(logs_mutex_);
  std::vector<LogEntry> logs;
  logs.swap(staged_logs_);
  return logs;
}

void RouteIngestor::run(RouteIngestConfig config) {
  const auto started_at = Clock::now();
  try {
    if (scheduler_ == nullptr) throw std::runtime_error("Route ingestor has no scheduler");
    RouteResolveResult resolved = resolve_route_segments(config.resolve);
    scheduler_->set_route_segments(resolved.segments);

    {
      RouteIngestStatus next = status();
      next.state = RouteIngestState::Loading;
      next.selection = resolved.selection;
      next.route_range = resolved.route_range;
      next.segments_resolved = resolved.segments.size();
      update_status(next);
    }

    const size_t worker_count = boundedWorkerCount(config.worker_count, resolved.segments.size());
    // Spawn bounded workers so each owns a local thread and a shared abort/state contract.
    std::vector<std::thread> workers;
      workers.reserve(worker_count);
      // first_segment_seen is atomic per run to set first_segment_seconds exactly once across workers.
      std::atomic<bool> first_segment_seen{false};

    auto worker = [&]() {
      // keep running until abort_ is set; each worker takes one segment at a time from scheduler.
      while (!abort_) {
        std::optional<SegmentWorkItem> work = scheduler_->take_next();
        if (!work.has_value()) return;
        try {
          SegmentLoadOptions options;
          options.extract.segment = work->segment;
          options.extract.coverage = work->range;
          options.local_cache = config.local_cache;
          SegmentLoadResult loaded = load_route_segment(*work, options, &abort_);
          if (abort_) {
            scheduler_->mark_pending(work->segment);
            return;
          }
          if (!loaded.car_fingerprint.empty()) {
            RouteIngestStatus next = status();
            if (next.car_fingerprint.empty()) {
              next.car_fingerprint = loaded.car_fingerprint;
              update_status(next);
            }
          }
          stage_timeline_spans(std::move(loaded.timeline_spans));
          stage_log_entries(std::move(loaded.logs));
          scheduler_->publish(std::move(loaded.batch));
          mutate_status(mark_loaded);
          mutate_status(markPublished);
          bool expected = false;
          if (first_segment_seen.compare_exchange_strong(expected, true)) {
            RouteIngestStatus next = status();
            next.first_segment_seconds = secondsSince(started_at);
            update_status(next);
          }
        } catch (const std::exception &err) {
          scheduler_->mark_failed(work->segment, err.what());
          mutate_status(mark_failed);
        }
      }
    };

    for (size_t i = 0; i < worker_count; ++i) workers.emplace_back(worker);
    for (std::thread &thread : workers) {
      if (thread.joinable()) thread.join();
    }

    RouteIngestStatus next = status();
    next.state = abort_ ? RouteIngestState::Canceled
                        : (next.segments_failed > 0 && next.segments_loaded == 0 ? RouteIngestState::Failed : RouteIngestState::Completed);
    next.total_seconds = secondsSince(started_at);
    if (next.state == RouteIngestState::Failed && next.error.empty()) {
      next.error = "All route segments failed to load";
    }
    update_status(next);
  } catch (const std::exception &err) {
    RouteIngestStatus next = status();
    next.state = abort_ ? RouteIngestState::Canceled : RouteIngestState::Failed;
    next.error = err.what();
    next.total_seconds = secondsSince(started_at);
    update_status(next);
  }
}

void RouteIngestor::update_status(const RouteIngestStatus &status) {
  // Serialize writes to status_ so every consumer sees a coherent snapshot.
  std::lock_guard lock(status_mutex_);
  status_ = status;
}

void RouteIngestor::mutate_status(void (*fn)(RouteIngestStatus *)) {
  // Serialize compound counter transitions so multiple workers don't clobber increments.
  std::lock_guard lock(status_mutex_);
  fn(&status_);
}

void RouteIngestor::stage_timeline_spans(std::vector<TimelineSpan> spans) {
  if (spans.empty()) return;
  // Hold timeline lock while appending so drain_timeline_spans() can't observe partial writes.
  std::lock_guard lock(timeline_mutex_);
  staged_timeline_spans_.insert(staged_timeline_spans_.end(),
                                std::make_move_iterator(spans.begin()),
                                std::make_move_iterator(spans.end()));
}

void RouteIngestor::stage_log_entries(std::vector<LogEntry> logs) {
  if (logs.empty()) return;
  // Hold logs lock while appending so drain_log_entries() can't observe partial writes.
  std::lock_guard lock(logs_mutex_);
  staged_logs_.insert(staged_logs_.end(),
                      std::make_move_iterator(logs.begin()),
                      std::make_move_iterator(logs.end()));
}

}  // namespace loggy
