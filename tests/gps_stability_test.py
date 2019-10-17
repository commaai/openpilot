#!/usr/bin/env python3

import os
import sys
import time
import random
import threading

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda, PandaSerial

INIT_GPS_BAUD = 9600
GPS_BAUD = 460800

def connect():
  pandas = Panda.list()
  print(pandas)

  # make sure two pandas are connected
  if len(pandas) != 2:
    print("Connect white and grey/black panda to run this test!")
    assert False

  # connect
  pandas[0] = Panda(pandas[0])
  pandas[1] = Panda(pandas[1])

  white_panda = None
  gps_panda = None

  # find out which one is white (for spamming the CAN buses)
  if pandas[0].is_white() and not pandas[1].is_white():
    white_panda = pandas[0]
    gps_panda = pandas[1]
  elif not pandas[0].is_white() and pandas[1].is_white():
    white_panda = pandas[1]
    gps_panda = pandas[0]
  else:
    print("Connect white and grey/black panda to run this test!")
    assert False
  return white_panda, gps_panda

def spam_buses_thread(panda):
  try:
    panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
    while True:
      at = random.randint(1, 2000)
      st = (b"test"+os.urandom(10))[0:8]
      bus = random.randint(0, 2)
      panda.can_send(at, st, bus)
  except Exception as e:
    print(e)

def read_can_thread(panda):
  try:
    while True:
      panda.can_recv()
  except Exception as e:
    print(e)

def init_gps(panda):
  def add_nmea_checksum(msg):
    d = msg[1:]
    cs = 0
    for i in d:
      cs ^= ord(i)
    return msg + "*%02X" % cs

  ser = PandaSerial(panda, 1, INIT_GPS_BAUD)

  # Power cycle the gps by toggling reset
  print("Resetting GPS")
  panda.set_esp_power(0)
  time.sleep(0.5)
  panda.set_esp_power(1)
  time.sleep(0.5)

  # Upping baud rate
  print("Upping GPS baud rate")
  msg = add_nmea_checksum("$PUBX,41,1,0007,0003,%d,0" % GPS_BAUD)+"\r\n"
  ser.write(msg)
  time.sleep(1)   # needs a wait for it to actually send

  # Reconnecting with the correct baud
  ser = PandaSerial(panda, 1, GPS_BAUD)

  # Sending all config messages boardd sends
  print("Sending config")
  ser.write("\xB5\x62\x06\x00\x14\x00\x03\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x1E\x7F")
  ser.write("\xB5\x62\x06\x3E\x00\x00\x44\xD2")
  ser.write("\xB5\x62\x06\x00\x14\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x35")
  ser.write("\xB5\x62\x06\x00\x14\x00\x01\x00\x00\x00\xC0\x08\x00\x00\x00\x08\x07\x00\x01\x00\x01\x00\x00\x00\x00\x00\xF4\x80")
  ser.write("\xB5\x62\x06\x00\x14\x00\x04\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1D\x85")
  ser.write("\xB5\x62\x06\x00\x00\x00\x06\x18")
  ser.write("\xB5\x62\x06\x00\x01\x00\x01\x08\x22")
  ser.write("\xB5\x62\x06\x00\x01\x00\x02\x09\x23")
  ser.write("\xB5\x62\x06\x00\x01\x00\x03\x0A\x24")
  ser.write("\xB5\x62\x06\x08\x06\x00\x64\x00\x01\x00\x00\x00\x79\x10")
  ser.write("\xB5\x62\x06\x24\x24\x00\x05\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5A\x63")
  ser.write("\xB5\x62\x06\x1E\x14\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3C\x37")
  ser.write("\xB5\x62\x06\x24\x00\x00\x2A\x84")
  ser.write("\xB5\x62\x06\x23\x00\x00\x29\x81")
  ser.write("\xB5\x62\x06\x1E\x00\x00\x24\x72")
  ser.write("\xB5\x62\x06\x01\x03\x00\x01\x07\x01\x13\x51")
  ser.write("\xB5\x62\x06\x01\x03\x00\x02\x15\x01\x22\x70")
  ser.write("\xB5\x62\x06\x01\x03\x00\x02\x13\x01\x20\x6C")

  print("Initialized GPS")

received_messages = 0
received_bytes = 0
send_something = False
def gps_read_thread(panda):
  global received_messages, received_bytes, send_something
  ser = PandaSerial(panda, 1, GPS_BAUD)
  while True:
    ret = ser.read(1024)
    time.sleep(0.001)
    l = len(ret)
    if l > 0:
      received_messages+=1
      received_bytes+=l
    if send_something:
      ser.write("test")
      send_something = False


CHECK_PERIOD = 5
MIN_BYTES = 10000
MAX_BYTES = 50000

min_failures = 0
max_failures = 0

if __name__ == "__main__":
  white_panda, gps_panda = connect()

  # Start spamming the CAN buses with the white panda. Also read the messages to add load on the GPS panda
  threading.Thread(target=spam_buses_thread, args=(white_panda,)).start()
  threading.Thread(target=read_can_thread, args=(gps_panda,)).start()

  # Start GPS checking
  init_gps(gps_panda)

  read_thread = threading.Thread(target=gps_read_thread, args=(gps_panda,))
  read_thread.start()
  while True:
    time.sleep(CHECK_PERIOD)
    if(received_bytes < MIN_BYTES):
      print("Panda is not sending out enough data! Got " + str(received_messages) + " (" + str(received_bytes) + "B) in the last " + str(CHECK_PERIOD) + " seconds")
      send_something = True
      min_failures+=1
    elif(received_bytes > MAX_BYTES):
      print("Panda is not sending out too much data! Got " + str(received_messages) + " (" + str(received_bytes) + "B) in the last " + str(CHECK_PERIOD) + " seconds")
      print("Probably not on the right baud rate, got reset somehow? Resetting...")
      max_failures+=1
      init_gps(gps_panda)
    else:
      print("Got " + str(received_messages) + " (" + str(received_bytes) + "B) messages in the last " + str(CHECK_PERIOD) + " seconds.")
      if(min_failures > 0):
        print("Total min failures: ", min_failures)
      if(max_failures > 0):
        print("Total max failures: ", max_failures)
    received_messages = 0
    received_bytes = 0



  