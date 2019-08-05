#!/usr/bin/env python
import sys
import binascii
from panda import Panda

def tesla_tester():

  try:
    print("Trying to connect to Panda over USB...")
    p = Panda()

  except AssertionError:
    print("USB connection failed. Trying WiFi...")

    try:
      p = Panda("WIFI")
    except:
      print("WiFi connection timed out. Please make sure your Panda is connected and try again.")
      sys.exit(0)

  body_bus_speed = 125 # Tesla Body busses (B, BF) are 125kbps, rest are 500kbps
  body_bus_num = 1 # My TDC to OBD adapter has PT on bus0 BDY on bus1 and CH on bus2
  p.set_can_speed_kbps(body_bus_num, body_bus_speed)

  # Now set the panda from its default of SAFETY_NOOUTPUT (read only) to SAFETY_ALLOUTPUT
  # Careful, as this will let us send any CAN messages we want (which could be very bad!)
  print("Setting Panda to output mode...")
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # BDY 0x248 is the MCU_commands message, which includes folding mirrors, opening the trunk, frunk, setting the cars lock state and more. For our test, we will edit the 3rd byte, which is MCU_lockRequest. 0x01 will lock, 0x02 will unlock:
  print("Unlocking Tesla...")
  p.can_send(0x248, "\x00\x00\x02\x00\x00\x00\x00\x00", body_bus_num)

  #Or, we can set the first byte, MCU_frontHoodCommand + MCU_liftgateSwitch, to 0x01 to pop the frunk, or 0x04 to open/close the trunk (0x05 should open both)
  print("Opening Frunk...")
  p.can_send(0x248, "\x01\x00\x00\x00\x00\x00\x00\x00", body_bus_num)

  #Back to safety...
  print("Disabling output on Panda...")
  p.set_safety_mode(Panda.SAFETY_NOOUTPUT)

  print("Reading VIN from 0x568. This is painfully slow and can take up to 3 minutes (1 minute per message; 3 messages needed for full VIN)...")

  vin = {}
  while True:
    #Read the VIN
    can_recv = p.can_recv()
    for address, _, dat, src  in can_recv:
      if src == body_bus_num:
        if address == 1384: #0x568 is VIN
          vin_index = int(binascii.hexlify(dat)[:2]) #first byte is the index, 00, 01, 02
          vin_string = binascii.hexlify(dat)[2:] #rest of the string is the actual VIN data
          vin[vin_index] = vin_string.decode("hex")
          print("Got VIN index " + str(vin_index) + " data " + vin[vin_index])
    #if we have all 3 parts of the VIN, print it and break out of our while loop
    if 0 in vin and 1 in vin and 2 in vin:
      print("VIN: " + vin[0] + vin[1] + vin[2][:3])
      break

if __name__ == "__main__":
  tesla_tester()
