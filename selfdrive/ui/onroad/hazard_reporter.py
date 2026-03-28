import time
import threading
import uuid

import requests

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.onroad.hazard_detection_metrics import metrics as comma1_metrics

BASE_URL = "https://roadpass.jpadams.xyz"
TIMEOUT = 10.0  # seconds


def _build_payload(sm, dongle_id: str, event_id: str, trigger_source: str) -> dict:
  car = sm['carState']
  gps = sm['gpsLocationExternal']
  sds = sm['selfdriveState']
  dms = sm['driverMonitoringState']
  ws = car.wheelSpeeds

  return {
    "event_id": event_id,
    "dongle_id": dongle_id,
    "detected_at_ms": int(time.time() * 1000),
    "trigger_source": trigger_source,

    "vehicle": {
      "speed_ms": car.vEgo,
      "accel_ms2": car.aEgo,
      "yaw_rate_rads": car.yawRate,
      "steering_angle_deg": car.steeringAngleDeg,
      "brake_pressed": car.brakePressed,
      "gear": car.gearShifter.raw,
      "wheel_speeds": {
        "fl": ws.fl,
        "fr": ws.fr,
        "rl": ws.rl,
        "rr": ws.rr,
      },
    },

    "location": {
      "latitude": gps.latitude,
      "longitude": gps.longitude,
      "altitude_m": gps.altitude,
      "bearing_deg": gps.bearingDeg,
      "speed_ms": gps.speed,
      "horizontal_accuracy_m": gps.horizontalAccuracy,
      "has_fix": gps.hasFix,
      "satellite_count": gps.satelliteCount,
      "gps_timestamp_ms": gps.unixTimestampMillis,
    },

    "openpilot": {
      "engaged": sds.enabled,
      "state": sds.state.raw,
      "experimental_mode": sds.experimentalMode,
    },

    "driver": {
      "face_detected": dms.faceDetected,
      "is_distracted": dms.isDistracted,
      "awareness_status": dms.awarenessStatus,
    },
  }


class HazardReporter:
  """
  Handles two-phase reporting of onroad hazard events.

  Phase 1 — POST /events: fired immediately at detection with a full vehicle
             snapshot. Runs on a daemon thread so the render loop is never blocked.

  Phase 2 — PATCH /events/response: fired when the driver answers the popup
             (or it times out). If the driver responds before Phase 1 returns,
             the response is queued and sent as soon as the event_id is known.
  """

  def __init__(self):
    self._params = Params()
    self._session = requests.Session()
    self._lock = threading.Lock()

    # Set during detect(), confirmed/overridden once POST returns.
    self._event_id: str | None = None
    # Holds (answer, latency_s) if respond() is called before POST returns.
    self._pending_response: tuple[str, float] | None = None

  def detect(self, sm, trigger_source: str = "bump_detector") -> None:
    """
    Snapshot the current SubMaster state and POST to /events.
    Call this at the moment of hazard detection.
    """
    dongle_id = self._params.get("DongleId") or "unknown"

    # Generate a client-side event_id immediately so respond() can always
    # send the PATCH even if the server is unreachable.
    event_id = str(uuid.uuid4())
    comma1_metrics.record_detect(trigger_source, event_id)
    payload = _build_payload(sm, dongle_id, event_id, trigger_source)

    with self._lock:
      self._event_id = None
      self._pending_response = None

    threading.Thread(target=self._post_event, args=(payload,), daemon=True).start()

  def respond(self, answer: str, latency_s: float) -> None:
    """
    Record the driver's response. Safe to call before the POST has returned —
    the PATCH will be queued and fired as soon as the event_id is available.

    answer: "yes" | "no" | "timeout"
    latency_s: seconds from popup appearance to response
    """
    comma1_metrics.record_driver_response(answer, latency_s)
    with self._lock:
      event_id = self._event_id
      if event_id is None:
        self._pending_response = (answer, latency_s)
        return

    threading.Thread(target=self._patch_response, args=(event_id, answer, latency_s), daemon=True).start()

  # ── private ────────────────────────────────────────────────────────────────

  def _post_event(self, payload: dict) -> None:
    try:
      resp = self._session.post(f"{BASE_URL}/events", json=payload, timeout=TIMEOUT)
      resp.raise_for_status()
      # Prefer the server's event_id if it sends one back, else keep ours.
      event_id = resp.json().get("event_id", payload["event_id"])
      cloudlog.info(f"HazardReporter: event created event_id={event_id}")
      comma1_metrics.record_post_result(True)
    except Exception as e:
      cloudlog.error(f"HazardReporter POST /events failed: {e}")
      comma1_metrics.record_post_result(False)
      event_id = payload["event_id"]

    with self._lock:
      self._event_id = event_id
      pending = self._pending_response
      self._pending_response = None

    if pending is not None:
      answer, latency_s = pending
      threading.Thread(target=self._patch_response, args=(event_id, answer, latency_s), daemon=True).start()

  def _patch_response(self, event_id: str, answer: str, latency_s: float) -> None:
    try:
      resp = self._session.patch(
        f"{BASE_URL}/events/response",
        json={
          "event_id": event_id,
          "response": answer,
          "response_latency_s": round(latency_s, 3),
        },
        timeout=TIMEOUT,
      )
      resp.raise_for_status()
      cloudlog.info(f"HazardReporter: response sent event_id={event_id} answer={answer}")
      comma1_metrics.record_patch_result(True)
    except Exception as e:
      cloudlog.error(f"HazardReporter PATCH /events/response failed: {e}")
      comma1_metrics.record_patch_result(False)
