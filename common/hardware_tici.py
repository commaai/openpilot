import serial

from common.hardware_base import HardwareBase
from cereal import log
import subprocess


NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength


def run_at_command(cmd, timeout=0.1):
  with serial.Serial("/dev/ttyUSB2", timeout=timeout) as ser:
    ser.write(cmd + b"\r\n")
    ser.readline()  # Modem echos request
    return ser.readline().decode().rstrip()


class Tici(HardwareBase):
  def get_sound_card_online(self):
    return True

  def get_imei(self, slot):
    if slot != 0:
      return ""

    for _ in range(10):
      try:
        imei = run_at_command(b"AT+CGSN")
        if len(imei) == 15:
          return imei
      except serial.SerialException:
        pass

    raise RuntimeError("Error getting IMEI")

  def get_serial(self):
    return self.get_cmdline()['androidboot.serialno']

  def get_subscriber_info(self):
    return ""

  def reboot(self, reason=None):
    subprocess.check_output(["sudo", "reboot"])

  def get_network_type(self):
    return NetworkType.wifi

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown
