#!/usr/bin/env python
import os
import serial
from selfdrive.locationd.test import ublox
import time
import datetime
import struct
import sys
from cereal import log
from common import realtime
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from selfdrive.locationd.test.ephemeris import EphemerisData, GET_FIELD_U

panda = os.getenv("PANDA") is not None   # panda directly connected
grey = not (os.getenv("EVAL") is not None)     # panda through boardd
debug = os.getenv("DEBUG") is not None   # debug prints
print_dB = os.getenv("PRINT_DB") is not None     # print antenna dB

timeout = 1
dyn_model = 4 # auto model
baudrate = 460800
ports = ["/dev/ttyACM0","/dev/ttyACM1"]
rate = 100 # send new data every 100ms

# which SV IDs we have seen and when we got iono
svid_seen = {}
svid_ephemeris = {}
iono_seen = 0

def configure_ublox(dev):
  # configure ports  and solution parameters and rate
  # TODO: configure constellations and channels to allow for 10Hz and high precision
  dev.configure_port(port=ublox.PORT_USB, inMask=1, outMask=1) # enable only UBX on USB
  dev.configure_port(port=0, inMask=0, outMask=0) # disable DDC

  if panda:
    payload = struct.pack('<BBHIIHHHBB', 1, 0, 0, 2240, baudrate, 1, 1, 0, 0, 0)
    dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_PRT, payload) # enable UART
  else:
    payload = struct.pack('<BBHIIHHHBB', 1, 0, 0, 2240, baudrate, 0, 0, 0, 0, 0)
    dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_PRT, payload) # disable UART

  dev.configure_port(port=4, inMask=0, outMask=0) # disable SPI
  dev.configure_poll_port()
  dev.configure_poll_port(ublox.PORT_SERIAL1)
  dev.configure_poll_port(ublox.PORT_SERIAL2)
  dev.configure_poll_port(ublox.PORT_USB)
  dev.configure_solution_rate(rate_ms=rate)

  # Configure solution
  payload = struct.pack('<HBBIIBB4H6BH6B', 5, 4, 3, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAV5, payload)
  payload = struct.pack('<B3BBB6BBB2BBB2B', 0, 0, 0, 0, 1,
                                            3, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ODO, payload)
  #payload = struct.pack('<HHIBBBBBBBBBBH6BBB2BH4B3BB', 0, 8192, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0)
  #dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAVX5, payload)

  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAV5)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAVX5)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ODO)

  # Configure RAW and PVT messages to be sent every solution cycle
  dev.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_PVT, 1)
  dev.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_RAW, 1)
  dev.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_SFRBX, 1)



def int_to_bool_list(num):
  # for parsing bool bytes
  return [bool(num & (1<<n)) for n in range(8)]


def gen_ephemeris(ephem_data):
  ephem = {'ephemeris':
          {'svId': ephem_data.svId,

           'toc': ephem_data.toc,
           'gpsWeek': ephem_data.gpsWeek,

           'af0': ephem_data.af0,
           'af1': ephem_data.af1,
           'af2': ephem_data.af2,

           'iode': ephem_data.iode,
           'crs': ephem_data.crs,
           'deltaN': ephem_data.deltaN,
           'm0': ephem_data.M0,

           'cuc': ephem_data.cuc,
           'ecc': ephem_data.ecc,
           'cus': ephem_data.cus,
           'a': ephem_data.A,

           'toe': ephem_data.toe,
           'cic': ephem_data.cic,
           'omega0': ephem_data.omega0,
           'cis': ephem_data.cis,

           'i0': ephem_data.i0,
           'crc': ephem_data.crc,
           'omega': ephem_data.omega,
           'omegaDot': ephem_data.omega_dot,

           'iDot': ephem_data.idot,

           'tgd': ephem_data.Tgd,

           'ionoCoeffsValid': ephem_data.ionoCoeffsValid,
           'ionoAlpha': ephem_data.ionoAlpha,
           'ionoBeta': ephem_data.ionoBeta}}
  return log.Event.new_message(ubloxGnss=ephem)


def gen_solution(msg):
  msg_data = msg.unpack()[0] # Solutions do not have any data in repeated blocks
  timestamp = int(((datetime.datetime(msg_data['year'],
                                      msg_data['month'],
                                      msg_data['day'],
                                      msg_data['hour'],
                                      msg_data['min'],
                                      msg_data['sec'])
                 - datetime.datetime(1970,1,1)).total_seconds())*1e+03
                 + msg_data['nano']*1e-06)
  gps_fix = {'bearing': msg_data['headMot']*1e-05,  # heading of motion in degrees
             'altitude': msg_data['height']*1e-03,  # altitude above ellipsoid
             'latitude': msg_data['lat']*1e-07,  # latitude in degrees
             'longitude': msg_data['lon']*1e-07,  # longitude in degrees
             'speed': msg_data['gSpeed']*1e-03,  # ground speed in meters
             'accuracy': msg_data['hAcc']*1e-03,  # horizontal accuracy (1 sigma?)
             'timestamp': timestamp,  # UTC time in ms since start of UTC stime
             'vNED': [msg_data['velN']*1e-03,
                     msg_data['velE']*1e-03,
                     msg_data['velD']*1e-03],  # velocity in NED frame in m/s
             'speedAccuracy': msg_data['sAcc']*1e-03,  # speed accuracy in m/s
             'verticalAccuracy': msg_data['vAcc']*1e-03,  # vertical accuracy in meters
             'bearingAccuracy': msg_data['headAcc']*1e-05,  # heading accuracy in degrees
             'source': 'ublox',
             'flags': msg_data['flags'],
  }
  return log.Event.new_message(gpsLocationExternal=gps_fix)

def gen_nav_data(msg, nav_frame_buffer):
  # TODO this stuff needs to be parsed and published.
  # refer to https://www.u-blox.com/sites/default/files/products/documents/u-blox8-M8_ReceiverDescrProtSpec_%28UBX-13003221%29.pdf
  # section 9.1
  msg_meta_data, measurements = msg.unpack()

  # parse GPS ephem
  gnssId = msg_meta_data['gnssId']
  if gnssId  == 0:
    svId =  msg_meta_data['svid']
    subframeId =  GET_FIELD_U(measurements[1]['dwrd'], 3, 8)
    words = []
    for m in measurements:
      words.append(m['dwrd'])

    # parse from
    if subframeId == 1:
      nav_frame_buffer[gnssId][svId] = {}
      nav_frame_buffer[gnssId][svId][subframeId] = words
    elif subframeId-1 in nav_frame_buffer[gnssId][svId]:
      nav_frame_buffer[gnssId][svId][subframeId] = words
    if len(nav_frame_buffer[gnssId][svId]) == 5:
      ephem_data = EphemerisData(svId, nav_frame_buffer[gnssId][svId])
      return gen_ephemeris(ephem_data)




def gen_raw(msg):
  # meta data is in first part of tuple
  # list of measurements is in second part
  msg_meta_data, measurements = msg.unpack()
  measurements_parsed = []
  for m in measurements:
    trackingStatus_bools = int_to_bool_list(m['trkStat'])
    trackingStatus = {'pseudorangeValid': trackingStatus_bools[0],
                      'carrierPhaseValid': trackingStatus_bools[1],
                      'halfCycleValid': trackingStatus_bools[2],
                      'halfCycleSubtracted': trackingStatus_bools[3]}
    measurements_parsed.append({
        'svId': m['svId'],
        'sigId': m['sigId'],
        'pseudorange': m['prMes'],
        'carrierCycles': m['cpMes'],
        'doppler': m['doMes'],
        'gnssId': m['gnssId'],
        'glonassFrequencyIndex': m['freqId'],
        'locktime': m['locktime'],
        'cno': m['cno'],
        'pseudorangeStdev': 0.01*(2**(m['prStdev'] & 15)), # weird scaling, might be wrong
        'carrierPhaseStdev': 0.004*(m['cpStdev'] & 15),
        'dopplerStdev': 0.002*(2**(m['doStdev'] & 15)), # weird scaling, might be wrong
        'trackingStatus': trackingStatus})
  if print_dB:
    cnos = {}
    for meas in measurements_parsed:
      cnos[meas['svId']] = meas['cno']
    print 'Carrier to noise ratio for each sat: \n', cnos, '\n'
  receiverStatus_bools = int_to_bool_list(msg_meta_data['recStat'])
  receiverStatus = {'leapSecValid': receiverStatus_bools[0],
                    'clkReset': receiverStatus_bools[2]}
  raw_meas = {'measurementReport': {'rcvTow': msg_meta_data['rcvTow'],
                'gpsWeek': msg_meta_data['week'],
                'leapSeconds': msg_meta_data['leapS'],
                'receiverStatus': receiverStatus,
                'numMeas': msg_meta_data['numMeas'],
                'measurements': measurements_parsed}}
  return log.Event.new_message(ubloxGnss=raw_meas)

def init_reader():
  port_counter = 0
  while True:
    try:
      dev = ublox.UBlox(ports[port_counter], baudrate=baudrate, timeout=timeout, panda=panda, grey=grey)
      configure_ublox(dev)
      return dev
    except serial.serialutil.SerialException as e:
      print(e)
      port_counter = (port_counter + 1)%len(ports)
      time.sleep(2)

def handle_msg(dev, msg, nav_frame_buffer):
  try:
    if debug:
      print(str(msg))
      sys.stdout.flush()
    if msg.name() == 'NAV_PVT':
      sol = gen_solution(msg)
      sol.logMonoTime = int(realtime.sec_since_boot() * 1e9)
      gpsLocationExternal.send(sol.to_bytes())
    elif msg.name() == 'RXM_RAW':
      raw = gen_raw(msg)
      raw.logMonoTime = int(realtime.sec_since_boot() * 1e9)
      ubloxGnss.send(raw.to_bytes())
    elif msg.name() == 'RXM_SFRBX':
      nav = gen_nav_data(msg, nav_frame_buffer)
      if nav is not None:
        nav.logMonoTime = int(realtime.sec_since_boot() * 1e9)
        ubloxGnss.send(nav.to_bytes())

    else:
      print "UNKNNOWN MESSAGE:", msg.name()
  except ublox.UBloxError as e:
    print(e)

  #if dev is not None and dev.dev is not None:
  #  dev.close()

def main(gctx=None):
  global gpsLocationExternal, ubloxGnss
  nav_frame_buffer = {}
  nav_frame_buffer[0] = {}
  for i in xrange(1,33):
    nav_frame_buffer[0][i] = {}


  gpsLocationExternal = messaging.pub_sock(service_list['gpsLocationExternal'].port)
  ubloxGnss = messaging.pub_sock(service_list['ubloxGnss'].port)

  dev = init_reader()
  while True:
    try:
      msg = dev.receive_message()
    except serial.serialutil.SerialException as e:
      print(e)
      dev.close()
      dev = init_reader()
    if msg is not None:
      handle_msg(dev, msg, nav_frame_buffer)

if __name__ == "__main__":
  main()
