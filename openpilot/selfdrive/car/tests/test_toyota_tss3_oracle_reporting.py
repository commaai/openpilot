import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from openpilot.selfdrive.car import toyota_tss3_oracle_auto as auto
from openpilot.selfdrive.car.toyota_tss3_oracle_status import STATUS_SCHEMA, OracleStatus
from openpilot.selfdrive.ui.mici.layouts.settings import developer
from openpilot.selfdrive.ui.tss3_oracle_runner import (
  CANCEL_FILENAME,
  Tss3OracleRunner,
  request_cooperative_cancel,
)


NATIVE_CATCH_SCHEMA = "tss3-oracle-native-catch-v1"


def status_row(**overrides):
  return {
    "schema": STATUS_SCHEMA, "stage": "done", "title": "Bringup complete",
    "detail": "All checks passed.", "progress": 100, "done": True, "error": False,
    **overrides,
  }


class TestOracleStatus(unittest.TestCase):
  def test_completion_requires_zero_exit(self):
    for returncode in (0, 2):
      with self.subTest(returncode=returncode):
        report = OracleStatus()
        report.feed(json.dumps(status_row(stage="verifying", done=False, progress=70)))
        report.feed(json.dumps(status_row()))
        result = report.finish(returncode)
        self.assertIs(result["done"], returncode == 0)
        self.assertIs(result["error"], returncode != 0)

  def test_exit_without_completion_is_not_success(self):
    for rows in ([], [status_row(stage="verifying", done=False, progress=70)]):
      with self.subTest(rows=rows):
        report = OracleStatus()
        for row in rows:
          report.feed(json.dumps(row))
        result = report.finish(0)
        self.assertIs(result["error"], True)
        self.assertIs(result["done"], False)

  def test_first_error_wins_over_later_output_and_exit(self):
    error = status_row(stage="error", done=False, error=True, detail="Original backend failure.", progress=0)
    rows = [error, {**error, "detail": "Later failure."}, status_row()]
    for returncode in (0, 2):
      with self.subTest(returncode=returncode):
        report = OracleStatus()
        updates = [update for row in rows if (update := report.feed(json.dumps(row))) is not None]
        self.assertEqual(updates, [error])
        self.assertEqual(report.finish(returncode), error)

  def test_last_diagnostic_is_preserved_across_blank_lines(self):
    report = OracleStatus()
    report.feed("Earlier diagnostic")
    diagnostic = json.dumps({"error": "permission denied opening log file"})
    report.feed(diagnostic)
    report.feed(" \n")
    result = report.finish(1)
    self.assertIs(result["error"], True)
    self.assertIn(diagnostic, result["detail"])
    self.assertNotIn("Earlier diagnostic", result["detail"])

  def test_malformed_status_cannot_report_success(self):
    for overrides in (
      {"progress": "not-a-number"}, {"progress": None}, {"progress": True},
      {"done": "false"}, {"error": "false"}, {"done": True, "error": True},
      {"title": []}, {"stage": "verifying", "done": True},
    ):
      with self.subTest(overrides=overrides):
        report = OracleStatus()
        self.assertIsNone(report.feed(json.dumps(status_row(**overrides))))
        result = report.finish(0)
        self.assertIs(result["error"], True)
        self.assertIs(result["done"], False)


class TestOracleUiReporting(unittest.TestCase):
  def test_completion_is_not_displayed_before_process_exit(self):
    runner = Tss3OracleRunner(start=False)

    def wait_for_exit():
      status = runner.snapshot()
      self.assertEqual(status["stage"], "finishing")
      self.assertIs(status["done"], False)
      return 0

    runner._proc = Mock(stdout=io.StringIO(json.dumps(status_row()) + "\n"), wait=wait_for_exit)
    runner._reader()
    self.assertIs(runner.snapshot()["done"], True)

  def test_missing_kit_still_opens_actionable_error_page(self):
    page = object()
    with patch.object(developer.ui_state, "is_offroad", return_value=True), \
         patch.object(developer, "tool_available", return_value=False), \
         patch.object(developer, "Tss3OracleBringupPage", return_value=page), \
         patch.object(developer.gui_app, "push_widget") as push_widget:
      developer.DeveloperLayoutMici._on_tss3_oracle_bringup(object())
    push_widget.assert_called_once_with(page)


class TestOracleLifecycleGuards(unittest.TestCase):
  def test_manual_cancel_is_cooperative_file_request(self):
    with tempfile.TemporaryDirectory() as td:
      run_dir = Path(td) / "run"
      request_cooperative_cancel(run_dir)
      self.assertEqual((run_dir / CANCEL_FILENAME).read_text(encoding="utf-8"), "cancel\n")

  def test_native_catch_marker_is_claimed_exactly_once(self):
    marker = {
      "schema": NATIVE_CATCH_SCHEMA,
      "target": "TOYOTA_CAMRY_TSS3",
      "verdict": "programming_request_sent_after_exact_50_03",
      "pandad_wrapper_pid": 123,
      "ignition_monotonic_ns": 10,
      "first_extended_tx_monotonic_ns": 20,
      "positive_extended_monotonic_ns": 40,
      "positive_extended_frame_hex": "065003003201f400",
    }
    with tempfile.TemporaryDirectory() as td:
      path = Path(td) / "catch.json"
      path.write_text(json.dumps(marker), encoding="utf-8")

      claimed = auto._claim_native_catch(path)
      assert claimed is not None
      claimed_path, claimed_marker = claimed
      self.assertEqual(claimed_marker["pandad_wrapper_pid"], 123)
      self.assertFalse(path.exists())
      self.assertTrue(claimed_path.exists())

      # The rename already took it; nothing is left to claim a second time.
      self.assertIsNone(auto._claim_native_catch(path))

  def test_native_catch_ignores_foreign_files(self):
    with tempfile.TemporaryDirectory() as td:
      path = Path(td) / "catch.json"
      for data in (b"{", b"not json", json.dumps({"schema": "something-else", "pandad_wrapper_pid": 123}).encode()):
        with self.subTest(data=data):
          path.write_bytes(data)
          self.assertIsNone(auto._claim_native_catch(path))
          self.assertTrue(path.exists())


class TestOracleRearmReporting(unittest.TestCase):
  def test_worker_readiness_does_not_erase_previous_result(self):
    with patch.object(auto, "_stop_warm_worker"), \
         patch.object(auto, "_exit_requested", False), \
         patch.object(auto, "_warm_worker", None), \
         patch.object(auto, "oracle_kit_compatibility", return_value=(True, "")), \
         patch.object(auto, "WARM_WORKER_PATH", Mock(is_socket=Mock(return_value=True))), \
         patch.object(auto.subprocess, "Popen", return_value=Mock(poll=Mock(return_value=None))), \
         patch.object(auto, "_write_status") as write_status:
      self.assertTrue(auto._start_warm_worker(publish_ready_status=False))
      write_status.assert_not_called()
      self.assertTrue(auto._start_warm_worker())
      self.assertEqual(write_status.call_args.args[0], "armed")


if __name__ == "__main__":
  unittest.main()
