import os
import capnp
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck

from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator
from openpilot.tools.lib.logreader import LogReader

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "10"))


@given(st.data())
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=[HealthCheck.large_base_example])
def test_log_backwards_compatibility(schema_path, data):
  # capnp global parser needs to be cleaned up to avoid schema/struct ID conflicts
  capnp.cleanup_global_schema_parser()
  old_log = capnp.load(schema_path)
  capnp.cleanup_global_schema_parser()

  msgs_dicts = FuzzyGenerator.get_random_event_msg(data.draw, log_schema=old_log, events=old_log.Event.schema.union_fields, real_floats=True)
  msgs = [old_log.Event.new_message(**m).as_reader() for m in msgs_dicts]
  dat = b"".join(msg.as_builder().to_bytes() for msg in msgs)

  lr = list(LogReader.from_bytes(dat))
  assert len(lr) == len(msgs)
  # calling which() on a removed union type will raise an exception
  assert {m.which() for m in lr} == {msg.which() for msg in msgs}
