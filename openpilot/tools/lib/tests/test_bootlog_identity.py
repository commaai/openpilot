from openpilot.tools.lib.bootlog import Bootlog


def test_equal_timestamps_on_different_devices_are_distinct():
  path = "1111111111111111/2020-01-01--00-00-00"
  first = Bootlog(path)
  other = Bootlog(path.replace("1111111111111111", "2222222222222222"))
  assert first == Bootlog("https://example.com/" + path)
  assert first != other
  assert first < other
  assert not other < first
  assert first < Bootlog("0000000000000000/2020-01-02--00-00-00")
