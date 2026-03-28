"""
Comma 1 (detection → report → driver confirm) instrumentation for RoadPass.

- **cloudlog**: grep for `roadpass.comma1` in swaglog / journal.
- **JSON snapshot**: `/tmp/roadpass_comma1_metrics.json` (overwritten on each update)
  for quick USB/SSH inspection without parsing full logs.

Use these counts to tune bump thresholds and to estimate false positives
(no / timeout vs yes among drivers who saw the popup).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field

from openpilot.common.swaglog import cloudlog

METRICS_JSON_PATH = "/tmp/roadpass_comma1_metrics.json"
_LOG_PREFIX = "roadpass.comma1"


@dataclass
class Comma1MetricsSnapshot:
  session_start_mono: float
  bump_triggers: int = 0
  manual_triggers: int = 0
  detect_calls: int = 0
  responses_yes: int = 0
  responses_no: int = 0
  responses_timeout: int = 0
  post_events_ok: int = 0
  post_events_fail: int = 0
  patch_response_ok: int = 0
  patch_response_fail: int = 0
  last_bump_a_ego_ms2: float | None = None
  last_bump_jerk_ms3: float | None = None
  last_trigger_source: str | None = None
  last_client_event_id: str | None = None
  last_driver_answer: str | None = None
  last_answer_mono: float | None = None


@dataclass
class _Mutable:
  snapshot: Comma1MetricsSnapshot = field(default_factory=lambda: Comma1MetricsSnapshot(session_start_mono=time.monotonic()))


class HazardDetectionMetrics:
  """Thread-safe counters for Comma 1 quality; singleton via module `metrics`."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._m = _Mutable()

  def _persist(self) -> None:
    try:
      with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(self._m.snapshot), f, indent=2)
    except OSError:
      pass

  def _log_summary(self, tag: str) -> None:
    with self._lock:
      s = self._m.snapshot
      answered = s.responses_yes + s.responses_no + s.responses_timeout
      yes_rate = (s.responses_yes / answered) if answered else None
      bump, manual = s.bump_triggers, s.manual_triggers
      dc = s.detect_calls
      ry, rn, rt = s.responses_yes, s.responses_no, s.responses_timeout
      pok, pfl = s.post_events_ok, s.post_events_fail
      xok, xfl = s.patch_response_ok, s.patch_response_fail
    yr = f"{yes_rate:.3f}" if yes_rate is not None else "n/a"
    cloudlog.info(
      f"{_LOG_PREFIX} {tag} bump={bump} manual={manual} "
      f"detect={dc} yes={ry} no={rn} "
      f"timeout={rt} post_ok={pok} post_fail={pfl} "
      f"patch_ok={xok} patch_fail={xfl} "
      f"yes_rate={yr}",
    )

  def record_bump_trigger(self, diag: dict[str, float]) -> None:
    with self._lock:
      s = self._m.snapshot
      s.bump_triggers += 1
      n = s.bump_triggers
      s.last_bump_a_ego_ms2 = diag.get("aEgoMs2")
      s.last_bump_jerk_ms3 = diag.get("jerkMs3")
      self._persist()
    cloudlog.info(
      f"{_LOG_PREFIX} bump_trigger n={n} "
      f"aEgo_ms2={diag.get('aEgoMs2')} jerk_ms3={diag.get('jerkMs3')} window_s={diag.get('windowS')}",
    )

  def record_manual_trigger(self) -> None:
    with self._lock:
      s = self._m.snapshot
      s.manual_triggers += 1
      n = s.manual_triggers
      self._persist()
    cloudlog.info(f"{_LOG_PREFIX} manual_trigger n={n}")

  def record_detect(self, trigger_source: str, client_event_id: str) -> None:
    with self._lock:
      s = self._m.snapshot
      s.detect_calls += 1
      dn = s.detect_calls
      s.last_trigger_source = trigger_source
      s.last_client_event_id = client_event_id
      self._persist()
    cloudlog.info(
      f"{_LOG_PREFIX} detect trigger={trigger_source} event_id={client_event_id} detect_n={dn}",
    )

  def record_driver_response(self, answer: str, latency_s: float) -> None:
    with self._lock:
      s = self._m.snapshot
      if answer == "yes":
        s.responses_yes += 1
      elif answer == "no":
        s.responses_no += 1
      else:
        s.responses_timeout += 1
      s.last_driver_answer = answer
      s.last_answer_mono = time.monotonic()
      self._persist()
    self._log_summary(f"response answer={answer} latency_s={latency_s:.2f}")

  def record_post_result(self, ok: bool) -> None:
    with self._lock:
      s = self._m.snapshot
      if ok:
        s.post_events_ok += 1
      else:
        s.post_events_fail += 1
      self._persist()
    cloudlog.info(f"{_LOG_PREFIX} post_events ok={ok} totals ok={self._m.snapshot.post_events_ok} fail={self._m.snapshot.post_events_fail}")

  def record_patch_result(self, ok: bool) -> None:
    with self._lock:
      s = self._m.snapshot
      if ok:
        s.patch_response_ok += 1
      else:
        s.patch_response_fail += 1
      self._persist()
    cloudlog.info(
      f"{_LOG_PREFIX} patch_response ok={ok} totals ok={self._m.snapshot.patch_response_ok} fail={self._m.snapshot.patch_response_fail}",
    )


metrics = HazardDetectionMetrics()
