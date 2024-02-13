#!/usr/bin/env python3
import time
from smbus2 import SMBus
from collections import namedtuple
from typing import List

# https://datasheets.maximintegrated.com/en/ds/MAX98089.pdf

AmpConfig = namedtuple('AmpConfig', ['name', 'value', 'register', 'offset', 'mask'])
EQParams = namedtuple('EQParams', ['K', 'k1', 'k2', 'c1', 'c2'])

def configs_from_eq_params(base, eq_params):
  return [
    AmpConfig("K (high)", (eq_params.K >> 8), base, 0, 0xFF),
    AmpConfig("K (low)", (eq_params.K & 0xFF), base + 1, 0, 0xFF),
    AmpConfig("k1 (high)", (eq_params.k1 >> 8), base + 2, 0, 0xFF),
    AmpConfig("k1 (low)", (eq_params.k1 & 0xFF), base + 3, 0, 0xFF),
    AmpConfig("k2 (high)", (eq_params.k2 >> 8), base + 4, 0, 0xFF),
    AmpConfig("k2 (low)", (eq_params.k2 & 0xFF), base + 5, 0, 0xFF),
    AmpConfig("c1 (high)", (eq_params.c1 >> 8), base + 6, 0, 0xFF),
    AmpConfig("c1 (low)", (eq_params.c1 & 0xFF), base + 7, 0, 0xFF),
    AmpConfig("c2 (high)", (eq_params.c2 >> 8), base + 8, 0, 0xFF),
    AmpConfig("c2 (low)", (eq_params.c2 & 0xFF), base + 9, 0, 0xFF),
  ]

BASE_CONFIG = [
  AmpConfig("MCLK prescaler", 0b01, 0x10, 4, 0b00110000),
  AmpConfig("PM: enable speakers", 0b11, 0x4D, 4, 0b00110000),
  AmpConfig("PM: enable DACs", 0b11, 0x4D, 0, 0b00000011),
  AmpConfig("Enable PLL1", 0b1, 0x12, 7, 0b10000000),
  AmpConfig("Enable PLL2", 0b1, 0x1A, 7, 0b10000000),
  AmpConfig("DAI1: I2S mode", 0b00100, 0x14, 2, 0b01111100),
  AmpConfig("DAI2: I2S mode", 0b00100, 0x1C, 2, 0b01111100),
  AmpConfig("DAI1 Passband filtering: music mode", 0b1, 0x18, 7, 0b10000000),
  AmpConfig("DAI1 voice mode gain (DV1G)", 0b00, 0x2F, 4, 0b00110000),
  AmpConfig("DAI1 attenuation (DV1)", 0x0, 0x2F, 0, 0b00001111),
  AmpConfig("DAI2 attenuation (DV2)", 0x0, 0x31, 0, 0b00001111),
  AmpConfig("DAI2: DC blocking", 0b1, 0x20, 0, 0b00000001),
  AmpConfig("DAI2: High sample rate", 0b0, 0x20, 3, 0b00001000),
  AmpConfig("ALC enable", 0b1, 0x43, 7, 0b10000000),
  AmpConfig("ALC/excursion limiter release time", 0b101, 0x43, 4, 0b01110000),
  AmpConfig("ALC multiband enable", 0b1, 0x43, 3, 0b00001000),
  AmpConfig("DAI1 EQ enable", 0b0, 0x49, 0, 0b00000001),
  AmpConfig("DAI2 EQ clip detection disabled", 0b1, 0x32, 4, 0b00010000),
  AmpConfig("DAI2 EQ attenuation", 0x5, 0x32, 0, 0b00001111),
  AmpConfig("Excursion limiter upper corner freq", 0b100, 0x41, 4, 0b01110000),
  AmpConfig("Excursion limiter lower corner freq", 0b00, 0x41, 0, 0b00000011),
  AmpConfig("Excursion limiter threshold", 0b000, 0x42, 0, 0b00001111),
  AmpConfig("Distortion limit (THDCLP)", 0x6, 0x46, 4, 0b11110000),
  AmpConfig("Distortion limiter release time constant", 0b0, 0x46, 0, 0b00000001),
  AmpConfig("Right DAC input mixer: DAI1 left", 0b0, 0x22, 3, 0b00001000),
  AmpConfig("Right DAC input mixer: DAI1 right", 0b0, 0x22, 2, 0b00000100),
  AmpConfig("Right DAC input mixer: DAI2 left", 0b1, 0x22, 1, 0b00000010),
  AmpConfig("Right DAC input mixer: DAI2 right", 0b0, 0x22, 0, 0b00000001),
  AmpConfig("DAI1 audio port selector", 0b10, 0x16, 6, 0b11000000),
  AmpConfig("DAI2 audio port selector", 0b01, 0x1E, 6, 0b11000000),
  AmpConfig("Enable left digital microphone", 0b1, 0x48, 5, 0b00100000),
  AmpConfig("Enable right digital microphone", 0b1, 0x48, 4, 0b00010000),
  AmpConfig("Enhanced volume smoothing disabled", 0b0, 0x49, 7, 0b10000000),
  AmpConfig("Volume adjustment smoothing disabled", 0b0, 0x49, 6, 0b01000000),
  AmpConfig("Zero-crossing detection disabled", 0b0, 0x49, 5, 0b00100000),
]

CONFIGS = {
  "tici": [
    AmpConfig("Right speaker output from right DAC", 0b1, 0x2C, 0, 0b11111111),
    AmpConfig("Right Speaker Mixer Gain", 0b00, 0x2D, 2, 0b00001100),
    AmpConfig("Right speaker output volume", 0x1c, 0x3E, 0, 0b00011111),
    AmpConfig("DAI2 EQ enable", 0b1, 0x49, 1, 0b00000010),

    *configs_from_eq_params(0x84, EQParams(0x274F, 0xC0FF, 0x3BF9, 0x0B3C, 0x1656)),
    *configs_from_eq_params(0x8E, EQParams(0x1009, 0xC6BF, 0x2952, 0x1C97, 0x30DF)),
    *configs_from_eq_params(0x98, EQParams(0x0F75, 0xCBE5, 0x0ED2, 0x2528, 0x3E42)),
    *configs_from_eq_params(0xA2, EQParams(0x091F, 0x3D4C, 0xCE11, 0x1266, 0x2807)),
    *configs_from_eq_params(0xAC, EQParams(0x0A9E, 0x3F20, 0xE573, 0x0A8B, 0x3A3B)),
  ],
  "tizi": [
    AmpConfig("Left speaker output from left DAC", 0b1, 0x2B, 0, 0b11111111),
    AmpConfig("Right speaker output from right DAC", 0b1, 0x2C, 0, 0b11111111),
    AmpConfig("Left Speaker Mixer Gain", 0b00, 0x2D, 0, 0b00000011),
    AmpConfig("Right Speaker Mixer Gain", 0b00, 0x2D, 2, 0b00001100),
    AmpConfig("Left speaker output volume", 0x17, 0x3D, 0, 0b00011111),
    AmpConfig("Right speaker output volume", 0x17, 0x3E, 0, 0b00011111),

    AmpConfig("DAI2 EQ enable", 0b0, 0x49, 1, 0b00000010),
    AmpConfig("DAI2: DC blocking", 0b0, 0x20, 0, 0b00000001),
    AmpConfig("ALC enable", 0b0, 0x43, 7, 0b10000000),
    AmpConfig("DAI2 EQ attenuation", 0x2, 0x32, 0, 0b00001111),
    AmpConfig("Excursion limiter upper corner freq", 0b001, 0x41, 4, 0b01110000),
    AmpConfig("Excursion limiter threshold", 0b100, 0x42, 0, 0b00001111),
    AmpConfig("Distortion limit (THDCLP)", 0x0, 0x46, 4, 0b11110000),
    AmpConfig("Distortion limiter release time constant", 0b1, 0x46, 0, 0b00000001),
    AmpConfig("Left DAC input mixer: DAI1 left", 0b0, 0x22, 7, 0b10000000),
    AmpConfig("Left DAC input mixer: DAI1 right", 0b0, 0x22, 6, 0b01000000),
    AmpConfig("Left DAC input mixer: DAI2 left", 0b1, 0x22, 5, 0b00100000),
    AmpConfig("Left DAC input mixer: DAI2 right", 0b0, 0x22, 4, 0b00010000),
    AmpConfig("Right DAC input mixer: DAI2 left", 0b0, 0x22, 1, 0b00000010),
    AmpConfig("Right DAC input mixer: DAI2 right", 0b1, 0x22, 0, 0b00000001),
    AmpConfig("Volume adjustment smoothing disabled", 0b1, 0x49, 6, 0b01000000),
  ],
}

class Amplifier:
  AMP_I2C_BUS = 0
  AMP_ADDRESS = 0x10

  def __init__(self, debug=False):
    self.debug = debug

  def _get_shutdown_config(self, amp_disabled: bool) -> AmpConfig:
    return AmpConfig("Global shutdown", 0b0 if amp_disabled else 0b1, 0x51, 7, 0b10000000)

  def _set_configs(self, configs: List[AmpConfig]) -> None:
    with SMBus(self.AMP_I2C_BUS) as bus:
      for config in configs:
        if self.debug:
          print(f"Setting \"{config.name}\" to {config.value}:")

        old_value = bus.read_byte_data(self.AMP_ADDRESS, config.register, force=True)
        new_value = (old_value & (~config.mask)) | ((config.value << config.offset) & config.mask)
        bus.write_byte_data(self.AMP_ADDRESS, config.register, new_value, force=True)

        if self.debug:
          print(f"  Changed {hex(config.register)}: {hex(old_value)} -> {hex(new_value)}")

  def set_configs(self, configs: List[AmpConfig]) -> bool:
    # retry in case panda is using the amp
    tries = 15
    for i in range(15):
      try:
        self._set_configs(configs)
        return True
      except OSError:
        print(f"Failed to set amp config, {tries - i - 1} retries left")
        time.sleep(0.02)
    return False

  def set_global_shutdown(self, amp_disabled: bool) -> bool:
    return self.set_configs([self._get_shutdown_config(amp_disabled), ])

  def initialize_configuration(self, model: str) -> bool:
    cfgs = [
      self._get_shutdown_config(True),
      *BASE_CONFIG,
      *CONFIGS[model],
      self._get_shutdown_config(False),
    ]
    return self.set_configs(cfgs)


if __name__ == "__main__":
  with open("/sys/firmware/devicetree/base/model") as f:
    model = f.read().strip('\x00')
  model = model.split('comma ')[-1]

  amp = Amplifier()
  amp.initialize_configuration(model)
