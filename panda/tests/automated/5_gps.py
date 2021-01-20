import time
from panda import PandaSerial
from .helpers import reset_pandas, test_all_gps_pandas, panda_connect_and_init

# Reset the pandas before running tests
def aaaa_reset_before_tests():
  reset_pandas()

@test_all_gps_pandas
@panda_connect_and_init
def test_gps_version(p):
  serial = PandaSerial(p, 1, 9600)
  # Reset and check twice to make sure the enabling works
  for i in range(2):
    # Reset GPS
    p.set_esp_power(0)
    time.sleep(2)
    p.set_esp_power(1)
    time.sleep(1)

    # Read startup message and check if version is contained
    dat = serial.read(0x1000)    # Read one full panda DMA buffer. This should include the startup message
    assert b'HPG 1.40ROV' in dat