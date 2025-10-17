from opendbc.car.values import PLATFORMS


class TestPlatformConfigs:
  def test_configs(self, subtests):

    for name, platform in PLATFORMS.items():
      with subtests.test(platform=str(platform)):
        assert platform.config._frozen

        if platform != "MOCK":
          assert len(platform.config.dbc_dict) > 0
        assert len(platform.config.platform_str) > 0

        assert name == platform.config.platform_str

        assert platform.config.specs is not None
