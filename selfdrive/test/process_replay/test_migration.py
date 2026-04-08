from openpilot.selfdrive.test.process_replay.migration import migrate_onroad_event_name


def test_migrate_onroad_event_name_maps_legacy_dm_events():
  assert migrate_onroad_event_name("preDriverDistracted") == "driverDistracted1"
  assert migrate_onroad_event_name("promptDriverDistracted") == "driverDistracted2"
  assert migrate_onroad_event_name("driverDistracted") == "driverDistracted3"
  assert migrate_onroad_event_name("preDriverUnresponsive") == "driverUnresponsive1"
  assert migrate_onroad_event_name("promptDriverUnresponsive") == "driverUnresponsive2"
  assert migrate_onroad_event_name("driverUnresponsive") == "driverUnresponsive3"


def test_migrate_onroad_event_name_preserves_other_events():
  assert migrate_onroad_event_name("tooDistracted") == "tooDistracted"
