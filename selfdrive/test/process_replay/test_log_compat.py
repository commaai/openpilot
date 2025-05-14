import os
import capnp
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from glob import glob

from cereal import CEREAL_PATH
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator
from openpilot.tools.lib.logreader import LogReader
from openpilot.common.run import run_cmd

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "2"))
TARGET_REMOTE = os.environ.get("TARGET_REMOTE", "origin")
TARGET_BRANCH = os.environ.get("TARGET_BRANCH", "master")


@pytest.fixture(scope="module")
def parent_schema_file(tmp_path_factory):
  tmp_dir = tmp_path_factory.mktemp("cereal")

  # TODO this will fail if the any capnp files are added/removed vs what was on the parent commit
  commit = run_cmd(["git", "merge-base", f"{TARGET_REMOTE}/{TARGET_BRANCH}", "HEAD"])
  for capnp_fp in glob(os.path.join(CEREAL_PATH, "**", "*.capnp"), recursive=True):
    fname = os.path.relpath(capnp_fp, CEREAL_PATH)
    capnp_url = f"https://raw.githubusercontent.com/commaai/openpilot/{commit}/cereal/{os.path.relpath(capnp_fp, CEREAL_PATH)}"
    tmp_capnp_path = tmp_dir / fname
    if not tmp_capnp_path.exists():
      run_cmd(["curl", "-o", str(tmp_capnp_path), "--create-dirs", capnp_url])

  return str(tmp_dir / "log.capnp")


@given(st.data())
@settings(max_examples=MAX_EXAMPLES, derandomize=True, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow])
def test_log_backwards_compatibility(parent_schema_file, data):
  # capnp global parser needs to be cleaned up to avoid schema/struct ID conflicts
  capnp_parser = capnp.SchemaParser()
  old_log = capnp_parser.load(os.path.abspath(parent_schema_file))

  msgs_dicts = FuzzyGenerator.get_random_event_msg(data.draw, log_schema=old_log, events=old_log.Event.schema.union_fields, real_floats=True)
  msgs = [old_log.Event.new_message(**m).as_reader() for m in msgs_dicts]
  dat = b"".join(msg.as_builder().to_bytes() for msg in msgs)

  lr = list(LogReader.from_bytes(dat))
  assert len(lr) == len(msgs)
  # calling which() on a removed union type will raise an exception
  assert {m.which() for m in lr} == {msg.which() for msg in msgs}
