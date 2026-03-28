import json

import openpilot.selfdrive.ui.onroad.hazard_detection_metrics as hdm


def test_metrics_json_roundtrip(tmp_path, monkeypatch):
  path = tmp_path / "m.json"
  monkeypatch.setattr(hdm, "METRICS_JSON_PATH", str(path))
  m = hdm.HazardDetectionMetrics()
  m.record_manual_trigger()
  m.record_driver_response("yes", 1.2)
  data = json.loads(path.read_text(encoding="utf-8"))
  assert data["manual_triggers"] == 1
  assert data["responses_yes"] == 1
  assert data["responses_no"] == 0
