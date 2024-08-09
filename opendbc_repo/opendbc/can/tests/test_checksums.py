from opendbc.can.parser import CANParser
from opendbc.can.packer import CANPacker


class TestCanChecksums:

  def test_honda_checksum(self):
    """Test checksums for Honda standard and extended CAN ids"""
    dbc_file = "honda_accord_2018_can_generated"
    msgs = [("LKAS_HUD", 0), ("LKAS_HUD_A", 0)]
    parser = CANParser(dbc_file, msgs, 0)
    packer = CANPacker(dbc_file)

    values = {
      'SET_ME_X41': 0x41,
      'STEERING_REQUIRED': 1,
      'SOLID_LANES': 1,
      'BEEP': 0,
    }

    # known correct checksums according to the above values
    checksum_std = [11, 10, 9, 8]
    checksum_ext = [4, 3, 2, 1]

    for std, ext in zip(checksum_std, checksum_ext):
      msgs = [
        packer.make_can_msg("LKAS_HUD", 0, values),
        packer.make_can_msg("LKAS_HUD_A", 0, values),
      ]
      parser.update_strings([0, msgs])

      assert parser.vl['LKAS_HUD']['CHECKSUM'] == std
      assert parser.vl['LKAS_HUD_A']['CHECKSUM'] == ext
