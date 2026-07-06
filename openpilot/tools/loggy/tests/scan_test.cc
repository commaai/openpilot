#include "catch2/catch.hpp"

#include "tools/loggy/backend/csv.h"
#include "tools/loggy/backend/scan.h"
#include "tools/loggy/backend/store.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace {

using loggy::CanEvent;
using loggy::MessageId;
using loggy::Store;
using loggy::StoreBatch;
using loggy::TimeRange;

// Stages one CAN event chunk per id and drains it so it is immediately queryable.
// Store isn't movable/copyable (it owns a mutex), so this fills a caller-owned instance.
void stageEvents(Store *store, std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id, TimeRange coverage) {
  StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {coverage};
  for (auto &[id, events] : per_id) {
    batch.can_events.push_back({.id = id, .range = coverage, .events = std::move(events), .segment = 0});
  }
  store->stage(std::move(batch));
  store->begin_frame();
}

}  // namespace

// -- (a) scan_find_bits_events: exact mismatch counts, ordering, and row cap. --

TEST_CASE("scan_find_bits_events computes exact mismatch counts and orders best-match first") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId perfect_match{.source = 1, .address = 0x10};
  const MessageId perfect_mismatch{.source = 1, .address = 0x20};

  std::vector<FindBitsEvent> events;
  // Every bit of 0xFF equals source_value=1: all 8 (byte,bit) rows for this address are 0% mismatch.
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_match, .data = {0xFF}});
  // Every bit of 0x00 differs from source_value=1: all 8 rows for this address are 100% mismatch.
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_mismatch, .data = {0x00}});

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 512;

  const std::vector<FindBitsRow> rows = loggy::scan_find_bits_events(events, params);
  REQUIRE(rows.size() == 16);

  // Best match (lowest mismatch percent) sorts first.
  CHECK(rows.front().address == perfect_match.address);
  CHECK(rows.front().byte_idx == 0);
  CHECK(rows.front().bit_idx == 0);
  CHECK(rows.front().total == 1);
  CHECK(rows.front().mismatches == 0);
  CHECK(rows.front().percent == Approx(0.0f));

  // Worst match sorts last.
  CHECK(rows.back().address == perfect_mismatch.address);
  CHECK(rows.back().byte_idx == 0);
  CHECK(rows.back().bit_idx == 7);
  CHECK(rows.back().total == 1);
  CHECK(rows.back().mismatches == 1);
  CHECK(rows.back().percent == Approx(100.0f));
}

TEST_CASE("scan_find_bits_events caps output at max_rows, keeping the best matches") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId perfect_match{.source = 1, .address = 0x10};
  const MessageId perfect_mismatch{.source = 1, .address = 0x20};
  std::vector<FindBitsEvent> events;
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_match, .data = {0xFF}});
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_mismatch, .data = {0x00}});

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 3;

  const std::vector<FindBitsRow> rows = loggy::scan_find_bits_events(events, params);
  REQUIRE(rows.size() == 3);
  for (const FindBitsRow &row : rows) {
    CHECK(row.address == perfect_match.address);
    CHECK(row.mismatches == 0);
  }
}

TEST_CASE("scan_find_bits_events drops rows at or below min_msgs") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;

  const MessageId target{.source = 1, .address = 0x30};
  std::vector<FindBitsEvent> events = {
    {.mono_time = 0.0, .source_value = 1, .id = target, .data = {0xFF}},
  };

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 1;  // requires total > 1, but this row only has total == 1
  params.max_rows = 512;

  CHECK(loggy::scan_find_bits_events(events, params).empty());
}

// -- (b) step_find_bits_job chunking: max_messages=1 must match one unbounded step. --

TEST_CASE("step_find_bits_job chunked by one message matches a single unbounded step") {
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId source_id{.source = 0, .address = 0x47};
  const MessageId target_a{.source = 0, .address = 0x100};
  const MessageId target_b{.source = 0, .address = 0x101};
  const MessageId target_c{.source = 0, .address = 0x102};
  const TimeRange range{0.0, 10.0};

  std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id;
  per_id.push_back({source_id, {
    {.mono_time = 1.0, .data = {0x01}},
    {.mono_time = 2.0, .data = {0x00}},
    {.mono_time = 3.0, .data = {0x01}},
  }});
  per_id.push_back({target_a, {
    {.mono_time = 1.0, .data = {0xFF}},
    {.mono_time = 2.0, .data = {0xFF}},
    {.mono_time = 3.0, .data = {0x00}},
  }});
  per_id.push_back({target_b, {
    {.mono_time = 1.5, .data = {0x00}},
    {.mono_time = 3.5, .data = {0xFF}},
  }});
  per_id.push_back({target_c, {
    {.mono_time = 2.5, .data = {0xAA}},
  }});

  Store store;
  stageEvents(&store, per_id, range);

  FindBitsParams params;
  params.range = range;
  params.source_bus = source_id.source;
  params.source_address = source_id.address;
  params.byte_idx = 0;
  params.bit_idx = 7;  // LSB of byte0, matches how the source data above was written (0x01 -> bit set)
  params.find_bus = 0;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 512;

  loggy::FindBitsJob unbounded_job = loggy::make_find_bits_job(store, params);
  REQUIRE(loggy::step_find_bits_job(unbounded_job, std::numeric_limits<size_t>::max()));
  REQUIRE(unbounded_job.done);

  loggy::FindBitsJob chunked_job = loggy::make_find_bits_job(store, params);
  int guard = 0;
  while (!loggy::step_find_bits_job(chunked_job, 1)) {
    REQUIRE(++guard < 1000);
  }
  REQUIRE(chunked_job.done);

  REQUIRE(chunked_job.rows.size() == unbounded_job.rows.size());
  for (size_t i = 0; i < unbounded_job.rows.size(); ++i) {
    const FindBitsRow &a = unbounded_job.rows[i];
    const FindBitsRow &b = chunked_job.rows[i];
    CHECK(a.address == b.address);
    CHECK(a.byte_idx == b.byte_idx);
    CHECK(a.bit_idx == b.bit_idx);
    CHECK(a.total == b.total);
    CHECK(a.mismatches == b.mismatches);
    CHECK(a.percent == b.percent);
  }
  // Sanity: the chunked run actually visited every target message, not zero.
  CHECK(chunked_job.id_index == chunked_job.ids.size());
  CHECK(chunked_job.ids.size() == 3);
}

// -- (a2) Find Bits parity audit regression: a row's total must be bounded by the candidate
// message's own event count (what a human counts scrubbing the Binary view for that id), not
// by how many times the source bit happened to sample it. Pairing source-event-driven (as the
// code used to) resamples "the last known target frame" once per source tick, so a slow
// candidate message racks up a total larger than its real event count -- a statistic nobody can
// reproduce by hand-checking the binary view. Pairing target-event-driven (each candidate frame
// against the source sample in effect at that time) bounds total by the candidate's own M
// events, however many K source transitions happened in between. --

TEST_CASE("step_find_bits_job bounds a row's total by the candidate's own event count, not the source's") {
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId source_id{.source = 0, .address = 0x47};
  const MessageId target_id{.source = 0, .address = 0x100};
  const TimeRange range{0.0, 10.0};

  // Source bit: K = 5 transitions (byte0 bit7: 1,0,1,0,1).
  std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id;
  per_id.push_back({source_id, {
    {.mono_time = 1.0, .data = {0x01}},
    {.mono_time = 2.0, .data = {0x00}},
    {.mono_time = 3.0, .data = {0x01}},
    {.mono_time = 4.0, .data = {0x00}},
    {.mono_time = 5.0, .data = {0x01}},
  }});
  // Candidate message: M = 2 events only, both 0xFF (every bit set).
  per_id.push_back({target_id, {
    {.mono_time = 1.2, .data = {0xFF}},  // preceded by source@1.0 (bit=1): every bit matches.
    {.mono_time = 4.8, .data = {0xFF}},  // preceded by source@4.0 (bit=0): every bit mismatches.
  }});

  Store store;
  stageEvents(&store, per_id, range);

  FindBitsParams params;
  params.range = range;
  params.source_bus = source_id.source;
  params.source_address = source_id.address;
  params.byte_idx = 0;
  params.bit_idx = 7;
  params.find_bus = 0;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 512;

  loggy::FindBitsJob job = loggy::make_find_bits_job(store, params);
  REQUIRE(loggy::step_find_bits_job(job, std::numeric_limits<size_t>::max()));
  REQUIRE(job.done);

  // Every (byte, bit) row for 0x100 is identical: both candidate frames are uniformly 0xFF.
  REQUIRE(job.rows.size() == 8);
  for (const FindBitsRow &row : job.rows) {
    CHECK(row.address == target_id.address);
    CHECK(row.byte_idx == 0);
    // Hand-computable: total == the candidate's own 2 events, never the source's 5 samples.
    CHECK(row.total == 2);
    CHECK(row.total <= 2);  // <= relevant event pairs (candidate's own event count), not K=5.
    CHECK(row.mismatches == 1);
    CHECK(row.percent == Approx(100.0f * 1.0f / 2.0f));
  }

  // Chunked-by-one must still agree with the unbounded run (chunking invariance untouched
  // by the pairing-direction fix).
  loggy::FindBitsJob chunked = loggy::make_find_bits_job(store, params);
  int guard = 0;
  while (!loggy::step_find_bits_job(chunked, 1)) REQUIRE(++guard < 1000);
  REQUIRE(chunked.rows.size() == job.rows.size());
  for (size_t i = 0; i < job.rows.size(); ++i) {
    CHECK(chunked.rows[i].total == job.rows[i].total);
    CHECK(chunked.rows[i].mismatches == job.rows[i].mismatches);
    CHECK(chunked.rows[i].percent == job.rows[i].percent);
  }
}

TEST_CASE("scan_find_bits_events: percent equals 100*mismatches/total, hand-computed on a 3-event set") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId target{.source = 0, .address = 0x50};
  // 3 events, same id: event 2 mismatches (source=1 vs data bit=0), events 1 and 3 match.
  std::vector<FindBitsEvent> events = {
    {.mono_time = 1.0, .source_value = 1, .id = target, .data = {0xFF}},  // match
    {.mono_time = 2.0, .source_value = 1, .id = target, .data = {0x00}},  // mismatch
    {.mono_time = 3.0, .source_value = 0, .id = target, .data = {0x00}},  // match
  };

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 512;

  const std::vector<FindBitsRow> rows = loggy::scan_find_bits_events(events, params);
  REQUIRE(rows.size() == 8);  // one row per bit of the single byte, all identical here
  for (const FindBitsRow &row : rows) {
    CHECK(row.total == 3);
    CHECK(row.mismatches == 1);
    CHECK(row.percent == Approx(100.0f / 3.0f));
    // The denominator behind percent must be the same total displayed in the row.
    CHECK(row.percent == Approx(100.0f * static_cast<float>(row.mismatches) / static_cast<float>(row.total)));
  }
}

// -- (c) find_signal_compare_value: table test for every comparator, incl. NaN -> false. --

TEST_CASE("find_signal_compare_value table for every comparator") {
  using loggy::FindSignalCompare;

  const double nan = std::numeric_limits<double>::quiet_NaN();

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::Any, 5.0));
  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::Any, 999.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Any, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::Equal, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::Equal, 5.1));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Equal, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::NotEqual, 5.1));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::NotEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::NotEqual, 5.0));

  CHECK(loggy::find_signal_compare_value(6.0, FindSignalCompare::Greater, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::Greater, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Greater, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::GreaterEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(4.9, FindSignalCompare::GreaterEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::GreaterEqual, 5.0));

  CHECK(loggy::find_signal_compare_value(4.0, FindSignalCompare::Less, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::Less, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Less, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::LessEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.1, FindSignalCompare::LessEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::LessEqual, 5.0));
}

// -- (d) find_signal job: finds a known planted 8-bit signal, no false positives when widening size range. --

TEST_CASE("find_signal job finds a planted 8-bit signal without false positives across a widened size range") {
  using loggy::FindSignalCompare;
  using loggy::FindSignalParams;

  const MessageId id{.source = 0, .address = 0x100};
  const TimeRange range{0.0, 10.0};
  // byte0 = 0xAB (171): the planted 8-bit unsigned LE value at start_bit 0.
  // byte1 = 0xFF: chosen so any wider/narrower window that spills into it changes the decoded
  // value (never coincidentally re-produces 171), proving there's no false positive.
  std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id;
  per_id.push_back({id, {{.mono_time = 1.0, .data = {0xAB, 0xFF}}}});
  Store store;
  stageEvents(&store, per_id, range);

  FindSignalParams params;
  params.range = range;
  params.buses = {0};
  params.addresses = {id.address};
  params.min_size = 1;
  params.max_size = 16;  // widened well past the true 8-bit signal
  params.little_endian = true;
  params.is_signed = false;
  params.factor = 1.0;
  params.offset = 0.0;
  params.target_value = 171.0;
  params.compare = FindSignalCompare::Equal;
  params.max_results = 512;

  loggy::FindSignalJob job = loggy::make_find_signal_job(store, params);
  int guard = 0;
  while (!loggy::step_find_signal_job(job, 32)) {
    REQUIRE(++guard < 1000);
  }

  REQUIRE(job.results.size() == 1);
  const auto &result = job.results.front();
  CHECK(result.id == id);
  CHECK(result.sig.start_bit == 0);
  CHECK(result.sig.size == 8);
  CHECK(result.msg_size == 2);
  REQUIRE(result.matches.size() == 1);
  CHECK(result.matches.front().second == Approx(171.0));
}

// -- (e) prepare_history_log_page: page-clamp, truncated flag, missing-signal doesn't crash. --

TEST_CASE("prepare_history_log_page clamps an out-of-range page index") {
  using loggy::HistoryLogParams;

  const MessageId id{.source = 0, .address = 0x200};
  const TimeRange range{0.0, 100.0};
  std::vector<CanEvent> events;
  for (int i = 0; i < 10; ++i) events.push_back({.mono_time = double(i), .data = {uint8_t(i)}});
  Store store;
  stageEvents(&store, {{id, events}}, range);

  HistoryLogParams params;
  params.page_size = 3;
  params.page_index = 99;  // far beyond page_count (4 pages for 10 rows at page_size 3)
  params.max_rows = 1000;

  const loggy::HistoryLogPage page = loggy::prepare_history_log_page(store, id, range, params);
  REQUIRE(page.page_count == 4);
  CHECK(page.page_index == 3);
  CHECK(page.total_rows == 10);
  CHECK(page.rows.size() == 1);  // last page holds the 1 remaining row (10 - 3*3)
  CHECK_FALSE(page.truncated);
}

TEST_CASE("prepare_history_log_page sets truncated when max_rows caps the match set") {
  using loggy::HistoryLogParams;

  const MessageId id{.source = 0, .address = 0x201};
  const TimeRange range{0.0, 100.0};
  std::vector<CanEvent> events;
  for (int i = 0; i < 10; ++i) events.push_back({.mono_time = double(i), .data = {uint8_t(i)}});
  Store store;
  stageEvents(&store, {{id, events}}, range);

  HistoryLogParams params;
  params.page_size = 100;
  params.page_index = 0;
  params.max_rows = 5;

  const loggy::HistoryLogPage page = loggy::prepare_history_log_page(store, id, range, params);
  CHECK(page.total_rows == 5);
  CHECK(page.truncated);
}

TEST_CASE("prepare_history_log_page with an unresolved compare signal filters everything, no crash") {
  using loggy::HistoryLogParams;

  const MessageId id{.source = 0, .address = 0x202};
  const TimeRange range{0.0, 100.0};
  std::vector<CanEvent> events = {
    {.mono_time = 1.0, .data = {0x01, 0x02}},
    {.mono_time = 2.0, .data = {0x03, 0x04}},
  };
  Store store;
  stageEvents(&store, {{id, events}}, range);

  HistoryLogParams params;
  params.compare_enabled = true;
  params.compare_signal = "does_not_exist";
  params.max_rows = 1000;
  params.page_size = 250;

  // msg == nullptr here (no DBC loaded) is the "missing signal" case: get_value can never be
  // called, but the page must still build cleanly instead of crashing.
  const loggy::HistoryLogPage page = loggy::prepare_history_log_page(store, id, range, params, nullptr);
  CHECK(page.total_rows == 0);
  CHECK(page.rows.empty());
  CHECK(page.page_count == 1);
  CHECK(page.page_index == 0);
}

// -- (f) summarize_message_events: bounding the range to the playhead, not the full route. --
// This is the Messages pane's live-table fix (REVIEW.md defect A): the caller passes
// {route_start, tracker_time}, not the route's full span, so the summary reflects only events
// seen up to the playhead.

TEST_CASE("summarize_message_events bounded at an early tracker time shows the first event only") {
  const MessageId id{.source = 0, .address = 0x300};
  const TimeRange route_range{0.0, 100.0};
  const std::vector<CanEvent> events = {
    {.mono_time = 10.0, .data = {0xAA, 0x01}},
    {.mono_time = 50.0, .data = {0xBB, 0x02}},
  };
  Store store;
  stageEvents(&store, {{id, events}}, route_range);

  const TimeRange up_to_tracker{route_range.start_, 30.0};
  const loggy::MessageSummary summary = loggy::summarize_message_events(store, id, up_to_tracker);
  CHECK(summary.count == 1);
  CHECK(summary.last_time == Approx(10.0));
  REQUIRE(summary.latest_data.size() == 2);
  CHECK(summary.latest_data[0] == 0xAA);
  CHECK(summary.latest_data[1] == 0x01);
}

TEST_CASE("summarize_message_events bounded past both events shows the latest one, not route-final") {
  const MessageId id{.source = 0, .address = 0x301};
  const TimeRange route_range{0.0, 100.0};
  const std::vector<CanEvent> events = {
    {.mono_time = 10.0, .data = {0xAA, 0x01}},
    {.mono_time = 50.0, .data = {0xBB, 0x02}},
  };
  Store store;
  stageEvents(&store, {{id, events}}, route_range);

  const TimeRange up_to_tracker{route_range.start_, 60.0};
  const loggy::MessageSummary summary = loggy::summarize_message_events(store, id, up_to_tracker);
  CHECK(summary.count == 2);
  CHECK(summary.last_time == Approx(50.0));
  REQUIRE(summary.latest_data.size() == 2);
  CHECK(summary.latest_data[0] == 0xBB);
  CHECK(summary.latest_data[1] == 0x02);
}

TEST_CASE("summarize_message_events paused at route start shows no events yet") {
  const MessageId id{.source = 0, .address = 0x302};
  const TimeRange route_range{0.0, 100.0};
  const std::vector<CanEvent> events = {
    {.mono_time = 10.0, .data = {0xAA}},
    {.mono_time = 50.0, .data = {0xBB}},
  };
  Store store;
  stageEvents(&store, {{id, events}}, route_range);

  const TimeRange up_to_tracker{route_range.start_, route_range.start_};
  const loggy::MessageSummary summary = loggy::summarize_message_events(store, id, up_to_tracker);
  CHECK(summary.count == 0);
  CHECK(summary.latest_data.empty());
}

// -- (g) can_message_csv: exported rows must belong to the requested id only (REVIEW.md
// defect #24 — History's "Save Msg" once exported 92k rows with zero rows of the selected
// message because a sibling pane silently reassigned the shared selection; see
// panes/messages.cc's selection-repair guard). Every row's bus/address column must match the
// requested id, regardless of how many other ids share the store.

namespace {

// One column per comma; none of the payloads below produce a hex/decoded field with a comma,
// so a plain split is exact (can_message_csv would otherwise quote such a field).
std::vector<std::vector<std::string>> parseCsvDataRows(const std::string &csv) {
  std::vector<std::vector<std::string>> rows;
  size_t line_start = csv.find('\n');
  if (line_start == std::string::npos) return rows;
  ++line_start;
  while (line_start < csv.size()) {
    size_t line_end = csv.find('\n', line_start);
    if (line_end == std::string::npos) line_end = csv.size();
    std::string line = csv.substr(line_start, line_end - line_start);
    if (!line.empty()) {
      std::vector<std::string> cells;
      size_t cell_start = 0;
      while (cell_start <= line.size()) {
        size_t comma = line.find(',', cell_start);
        if (comma == std::string::npos) comma = line.size();
        cells.push_back(line.substr(cell_start, comma - cell_start));
        cell_start = comma + 1;
      }
      rows.push_back(std::move(cells));
    }
    line_start = line_end + 1;
  }
  return rows;
}

}  // namespace

TEST_CASE("can_message_csv exports only the requested id's rows, even with other ids in the store") {
  const MessageId target{.source = 0, .address = 0x7E};
  const MessageId other_a{.source = 0, .address = 0x123};
  const MessageId other_b{.source = 1, .address = 0x7E};  // same address, different bus — must not leak in either.
  const TimeRange route_range{0.0, 100.0};

  const std::vector<CanEvent> target_events = {
    {.mono_time = 1.0, .bus_time = 5, .data = {0x01}},
    {.mono_time = 2.0, .bus_time = 6, .data = {0x02}},
  };
  const std::vector<CanEvent> other_a_events = {
    {.mono_time = 1.5, .bus_time = 7, .data = {0xAA}},
    {.mono_time = 2.5, .bus_time = 8, .data = {0xBB}},
    {.mono_time = 3.5, .bus_time = 9, .data = {0xCC}},
  };
  const std::vector<CanEvent> other_b_events = {
    {.mono_time = 1.2, .bus_time = 10, .data = {0xDD}},
  };

  Store store;
  stageEvents(&store, {{target, target_events}, {other_a, other_a_events}, {other_b, other_b_events}}, route_range);

  const std::string csv = loggy::can_message_csv(store, target, route_range);
  const std::vector<std::vector<std::string>> rows = parseCsvDataRows(csv);
  REQUIRE(rows.size() == target_events.size());
  for (const std::vector<std::string> &row : rows) {
    REQUIRE(row.size() == 7);
    CHECK(row[2] == "0");     // bus column
    CHECK(row[3] == "0x7E");  // address column
  }

  // Sibling ids (including the same address on a different bus) must not appear at all.
  const std::string other_a_csv = loggy::can_message_csv(store, other_a, route_range);
  for (const std::vector<std::string> &row : parseCsvDataRows(other_a_csv)) {
    CHECK(row[3] == "0x123");
  }
  const std::string other_b_csv = loggy::can_message_csv(store, other_b, route_range);
  REQUIRE(parseCsvDataRows(other_b_csv).size() == other_b_events.size());
  for (const std::vector<std::string> &row : parseCsvDataRows(other_b_csv)) {
    CHECK(row[2] == "1");
    CHECK(row[3] == "0x7E");
  }
}
