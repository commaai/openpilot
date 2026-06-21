from openpilot.common.hardware.tici.modem import Modem


def test_read_cellular_dns_uses_modem_reported_servers(monkeypatch):
  modem = Modem()
  monkeypatch.setattr(modem, "_atv", lambda *_: '1,5,"foo_bar_apn","10.0.0.1","10.0.0.2","9.9.9.9","149.112.112.112"')

  assert modem._read_cellular_dns() == ["9.9.9.9", "149.112.112.112"]


def test_read_cellular_dns_falls_back_when_modem_omits_servers(monkeypatch):
  modem = Modem()
  monkeypatch.setattr(modem, "_atv", lambda *_: '1,5,"foo_bar_apn","26.83.241.123"')

  assert modem._read_cellular_dns() == ["8.8.8.8", "1.1.1.1"]
