from openpilot.selfdrive.car.gwm.gwmcan import gwm_crc
from .mocks import GWM_HAVAL_CRC_SAMPLE_CSV, haval_crc_sample_to_array

def test_gwm_crc():
  crc_samples = haval_crc_sample_to_array(GWM_HAVAL_CRC_SAMPLE_CSV)

  def array_to_bytes(array: list) -> bytes:
    byte_array = bytearray(int(x) for x in array)
    return bytes(byte_array)

  for crc_sample in crc_samples:
    data = array_to_bytes(crc_sample[1:])
    crc = int(crc_sample[0])
    assert gwm_crc(data) == crc, "Invalid CRC value for data: " + str(data) + " Expected: " + str(crc) + " Got: " + str(gwm_crc(data))

  if len(crc_samples) == 0:
    raise AssertionError("No CRC samples found in GWM_HAVAL_CRC_SAMPLE_CSV.")
