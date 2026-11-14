from unittest import mock

from openpilot.common.hardware.comma import hardware
from openpilot.common.test import OpenpilotTestCase


def stock_ir_ma(percent: int) -> float:
  # led:torch_2 on the downstream kernel: openpilot writes mA, qpnp-flash-v2 rounds to its 12.5 mA steps, at least one
  ma = percent * 300 // 100
  if ma == 0:
    return 0
  return ((max(ma, 12) * 1000 + 6250) // 12500) * 12.5

def upstream_ir_ma(brightness: int) -> float:
  # ir:torch on the upstream kernel: leds-qcom-flash's floor(b * 300 / LED_FULL) mA, then its 5 mA steps
  if brightness == 0:
    return 0
  itarg = (brightness * 300 // 255) * 1000 + 1
  return (itarg // 5000 - 1 + 1 if itarg > 5000 else 1) * 5


class TestHardwareComma(OpenpilotTestCase):
  def test_ir_current_matches_stock(self):
    for percent in range(101):
      b = hardware.ir_torch_brightness(percent)
      assert 0 <= b <= 255
      # within half of one of the upstream driver's 5 mA steps of the downstream current
      assert abs(upstream_ir_ma(b) - stock_ir_ma(percent)) <= 2.5, percent
    assert hardware.ir_torch_brightness(100) == 255
    assert hardware.ir_torch_brightness(0) == 0
    assert hardware.ir_torch_brightness(-10) == 0
    assert hardware.ir_torch_brightness(200) == 255

  def test_hwmon(self):
    names = {"/sys/class/hwmon/hwmon0/name": "som_power\n"}
    hardware._hwmon.clear()
    with mock.patch("glob.glob", side_effect=lambda _: list(names)) as scan, \
         mock.patch("pathlib.Path.read_text", lambda p: names[str(p)]), mock.patch("time.monotonic", return_value=100.):
      assert str(hardware.hwmon("som_power")) == "/sys/class/hwmon/hwmon0"
      # a miss is scanned again only after 10 s
      assert hardware.hwmon("ina231") is None
      assert hardware.hwmon("ina231") is None
      assert scan.call_count == 2
    names["/sys/class/hwmon/hwmon1/name"] = "ina231\n"
    with mock.patch("glob.glob", side_effect=lambda _: list(names)), \
         mock.patch("pathlib.Path.read_text", lambda p: names[str(p)]), mock.patch("time.monotonic", return_value=110.):
      assert str(hardware.hwmon("ina231")) == "/sys/class/hwmon/hwmon1"
    hardware._hwmon.clear()

  def test_gpu_usage_percent(self):
    # 50 ms of busy time over 100 ms, from two clients. a new client counts from its next sample
    assert hardware.gpu_usage_percent(0, {"1": 0, "2": 100}, int(100e6), {"1": int(20e6), "2": int(30e6) + 100, "3": int(1e9)}) == 50.
    assert hardware.gpu_usage_percent(0, {"1": 0}, int(1e6), {"1": int(2e6)}) == 100.
    assert hardware.gpu_usage_percent(10, {}, 10, {}) == 0.

  def test_drm_engine_busy(self):
    fdinfo = "pos:\t0\nflags:\t02100002\ndrm-driver:\tmsm\ndrm-client-id:\t7\ndrm-engine-gpu:\t123456 ns\ndrm-cycles-gpu:\t99\n"
    with mock.patch("builtins.open", mock.mock_open(read_data=fdinfo)):
      assert hardware.drm_engine_busy("/proc/1/fdinfo/3") == {"7": 123456}
    with mock.patch("builtins.open", mock.mock_open(read_data="pos:\t0\nflags:\t0\n")):
      assert hardware.drm_engine_busy("/proc/1/fdinfo/3") == {}
    assert hardware.drm_engine_busy("/proc/does-not-exist/fdinfo/3") == {}

  def test_drm_fdinfo(self):
    # every DRM fd counts
    links = {"3": "/dev/dri/card0", "4": "/dev/dri/renderD128", "5": "socket:[123]", "6": "/dev/kgsl-3d0"}
    with mock.patch("os.listdir", return_value=list(links)), mock.patch("os.readlink", side_effect=lambda p: links[p.split("/")[-1]]):
      assert hardware.drm_fdinfo(10) == ["/proc/10/fdinfo/3", "/proc/10/fdinfo/4"]
    assert hardware.drm_fdinfo(-1) == []

  def test_gpu_usage_clients(self):
    hw = hardware.HardwareComma()
    with mock.patch.object(hardware, "drm_fdinfo", return_value=["/proc/10/fdinfo/5"]) as fdinfo, \
         mock.patch.object(hardware, "drm_engine_busy", side_effect=[{"1": 0}, {"1": int(1e9)}, {}, {"1": int(2e9)}]), \
         mock.patch("time.monotonic_ns", side_effect=[int(1e9), int(3e9), int(4e9), int(5e9)]):
      hw._drm_gpu_usage_percent([10])
      assert hw._drm_gpu_usage_percent([10]) == 50.
      # the fds of a known client are listed once
      assert fdinfo.call_count == 1
      # until one isn't a DRM fd any more
      hw._drm_gpu_usage_percent([10])
      hw._drm_gpu_usage_percent([10])
      assert fdinfo.call_count == 2
    # clients that went away are dropped
    hw._drm_gpu_usage_percent([])
    assert hw._drm_fdinfo == {}

  def test_mainline_thermal_zones(self):
    # the zone names in the upstream kernel's DT: sdm845.dtsi and pm8998.dtsi, and the board's in sdm845-comma-mici.dts
    soc = [f"cpu{i}-thermal" for i in range(8)] + ["gpu-top-thermal", "gpu-bottom-thermal", "q6-hvx-thermal", "mem-thermal",
                                                  "pm8998-thermal"]
    board = ["intake-thermal", "exhaust-thermal", "bottom-soc-thermal", "gnss-thermal"]
    for device, zones in (("mici", soc + board), ("tizi", soc)):
      hw = hardware.HardwareComma()
      with mock.patch.object(hardware, "upstream_kernel", return_value=True), \
           mock.patch.object(hw, "get_device_type", return_value=device):
        config = hw.get_thermal_config()
      names = [z.name for v in vars(config).values() if v is not None for z in (v if isinstance(v, list) else [v])]
      assert sorted(names) == sorted(zones)
      assert all(isinstance(z, hardware.MainlineThermalZone) for z in config.cpu)

  def test_mainline_thermal_zone_missing(self):
    # reads 0 like the downstream kernel's zones, and is logged once
    zone = hardware.MainlineThermalZone("intake-thermal")
    def fake_open(path, *args, **kwargs):
      if path.endswith("thermal_zone0/type"):
        return mock.mock_open(read_data="exhaust-thermal")()
      raise FileNotFoundError(path)
    with mock.patch("os.listdir", return_value=["thermal_zone0"]), mock.patch("builtins.open", side_effect=fake_open), \
         mock.patch("openpilot.common.swaglog.cloudlog.error") as error:
      assert zone.read() == 0
      assert zone.read() == 0
    assert error.call_count == 1
