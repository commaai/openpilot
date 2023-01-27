import time
import argparse
import struct
from datetime import datetime
from collections import defaultdict

import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes

MAX_SATS = 150

UBX_HEADER = b'\xb5\x62'
UBX_RXM_SFRBX = b'\x02\x13'
UBX_NAV_ORB = b'\x01\x34'
UBX_NAV_PVT = b'\x01\x07'
UBX_RAW_MEAS = b'\x02\x15'
UBX_MON_HW2 = b'\x0a\x0b'

RAW_EPHEM_TYPE = 1
RAW_NAV_ORB_TYPE = 2
RAW_NAV_PVT_TYPE = 3

GNSS_GPS = 0
GNSS_GLONASS = 6

def print_stats(start_time, ephem_stats, sat_meas, meas_cnt, locktime_per_sat, raw_ephem):
  print(f"----- {datetime.now().strftime('%H:%M:%S.%f')}, Runtime: {datetime.now() - start_time}")
  for sat in range(len(ephem_stats)):
    if ephem_stats[sat] != 0:
      print(f"{sat}: {round(ephem_stats[sat]/meas_cnt, 2)} ephem/min")

  print("Measurements (in last minute):")
  m6 = []
  for sat in sat_meas:
    if sat_meas[sat] in [60, 61, 600, 601]:
      m6.append(sat)
    else:
      print(f"{sat}: {sat_meas[sat]}")
  print(f"6club msgs sats: {m6}")

  print("Locktime per sat (in last minute):")
  for sat in locktime_per_sat:
    print(f"  {sat}: {locktime_per_sat[sat]}")

  print("Raw ephemeris per sat (total):")
  for sat in raw_ephem:
    print(f"  Sat {sat}:")
    for e in raw_ephem[sat]:
      print(f"    {e}")


def print_data(ephem_data):
  print(f"----- {datetime.now().strftime('%H:%M:%S.%f')}")
  for sat in ephem_data:
    print(f"{sat}: {ephem_data[sat]}")


def log_test_summary(num_sv, orb_db, ephem_data):
  print(f"UbloxGnss received Ephemeris: {len(ephem_data)}")
  for sat in ephem_data:
    print(f"{sat}: {ephem_data[sat]}")

  print(f"Num SV for fix: {num_sv}")
  print(f"Ublox sat db:")
  for sat in orb_db:
    gid = orb_db[sat][0]
    svFlags = orb_db[sat][1]
    eph = orb_db[sat][2]
    print(f"  {sat}: g: {gid} h:{svFlags&0x3} v:{svFlags>>2} u:{eph&0x1f} s:{eph>>5}")


def check_raw_ephemeris(raw_data):

  results = {
    RAW_EPHEM_TYPE: [],
    RAW_NAV_ORB_TYPE: [],
    RAW_NAV_PVT_TYPE: 0,
  }

  for em in raw_data.ubloxRaw.split(UBX_HEADER):
    if len(em) == 0 or em[:2] == UBX_RAW_MEAS or em[:2] == UBX_MON_HW2:
      continue

    if em[:2] == UBX_RXM_SFRBX:
      # parse easy to use data
      raw_data = em[4:] # skip length
      gnss_id = raw_data[0]
      sv_id = raw_data[1]
      num_words = raw_data[4]
      chn = raw_data[5]
      version = raw_data[6]
      ephem_raw = raw_data[8:8+num_words*4]
      #print(f"RAW Ephem: {sv_id} {gnss_id} {chn} {version} {num_words} {ephem_raw}")

      if gnss_id == GNSS_GPS:
        # tow counter is in word 2
        #print(f"ephem tow counter: {tow_counter}")
        '''
        For parsing ephemeris data
        ephem_raw = raw_data[8:8+num_words*4]
        for i in range(10):
          print(hex(struct.unpack("I", data[i*4:i*4+4])[0]>>6))
        # check ublox_msg.cc for proper parsing
        '''

        #print(f"GPS TLM: {hex(struct.unpack('I', ephem_raw[:4])[0]>>6)}")

        i = 1
        tow_counter = ((struct.unpack("I", ephem_raw[i*4:i*4+4])[0]>>6) >> 7) & 0x1FFFF
        subframe_id = ((struct.unpack("I", ephem_raw[i*4:i*4+4])[0]>>6) >> 2) & 0x7

        results[RAW_EPHEM_TYPE].append([sv_id, gnss_id, chn, version, tow_counter, subframe_id, ephem_raw])
      elif gnss_id == GNSS_GLONASS:
        if sv_id == 255:
          # data can be decoded before identifying the SV number, in this case 255
          # is returned, which means "unknown"  (ublox p32)
          continue

        words = struct.unpack("IIII", ephem_raw)
        string_number = (words[0]>>27)&0xF
        data80_54 = (words[0])&0x3ffffff
        data53_22 = (words[1])
        data21_9 = (words[2]>>19)
        superframe_number = (words[3]>>16)
        frame_number = (words[3]&0xFF)

        print(f"Glonass ephemeris: {[sv_id, gnss_id, chn, version, num_words, ephem_raw]}")
        print(f"  {[string_number, data80_54, data53_22, data21_9, superframe_number, frame_number]}")

    if em[:2] == UBX_NAV_ORB:
      orb_db = {}
      #print(f"NAV-ORB Database:")
      # parse orbit database
      orb_data = em[4:]
      num_sv = orb_data[5]
      for i in range(num_sv):
        sv_data = orb_data[8+i*6:8+i*6+6]
        gnssId = sv_data[0]
        svId = sv_data[1]
        svFlag = sv_data[2] # visibility and health
        eph = sv_data[3]    # ephemeris usability and source
        #alm = sv_data[4]    # almanac usabilite and source
        #orb = sv_data[5]    # not needed
        orb_db[svId] = [gnssId, svFlag, eph]
        #print(f"  {svId} {gnssId} h:{svFlag&0x3} v:{svFlag>>2} u:{eph&0x1f} s:{eph>>5}")
      results[RAW_NAV_ORB_TYPE].append(orb_db)

    if em[:2] == UBX_NAV_PVT:
      #print(f"NAV-PVT message: {em[4:]}")
      try:
        data = em[4:]
        flags = data[21]
        num_sv = data[23]
        #print(f"has_fix: {flags&1} num_sv: {num_sv}")
        if (flags&1) == 1:
          results[RAW_NAV_PVT_TYPE] = num_sv
      except:
        print(f"NAV-PVT crashed: {em}")

  return results


def drain_raw_sock(ublox_raw, raw_ephem, orb_dbs):
  for rm in messaging.drain_sock(ublox_raw):
    results = check_raw_ephemeris(rm)

    if len(results[RAW_EPHEM_TYPE]) > 0:
      for r in results[RAW_EPHEM_TYPE]:
        sv_id, _, chn, version, tow_counter, subframe_id, ephem_raw = r # [sv_id, gnss_id, chn, version, raw_data]
        raw_ephem[sv_id].append([chn, version, tow_counter, subframe_id, ephem_raw])

    if len(results[RAW_NAV_ORB_TYPE]) > 0:
      for r in results[RAW_NAV_ORB_TYPE]:
        orb_dbs.append(r)

    #if results[RAW_NAV_PVT_TYPE] > 0:
    #  return results[RAW_NAV_PVT_TYPE] # num_sv

  return 0


def drain_gnss_sock(ublox_gnss, sat_meas, locktime_per_sat, ephem_data):
  for msg in messaging.drain_sock(ublox_gnss):
    if msg.ubloxGnss.which() == "measurementReport":
      for m in msg.ubloxGnss.measurementReport.measurements:
        if m.gnssId == 6 and m.svId == 255:
          continue

        svId = m.svId + 100 if m.gnssId == 6 else m.svId
        sat_meas[svId] += 1
        locktime_per_sat[svId] = m.locktime

    if msg.ubloxGnss.which() == "ephemeris":
      ephem_data[msg.ubloxGnss.ephemeris.svId] += 1


def update_stats(ephem_data, ephem_stats, diff_ephems_per_min, total_ephems_per_min):
  for sat in ephem_data:
    ephem_stats[sat] += ephem_data[sat]
  diff_ephems_per_min.append(len(ephem_data))
  total_ephems_per_min.append(sum(n for n in ephem_data.values()))


def log_loop(stop_at_fix=False):
  ephem_stats = [0]*MAX_SATS
  sat_meas = defaultdict(int)
  meas_cnt = 0

  last_log = time.monotonic()
  start_time = datetime.now()
  orb_dbs = []

  ephem_data = defaultdict(int)
  total_ephems_per_min = []
  diff_ephems_per_min = []
  locktime_per_sat = {}
  raw_ephem = defaultdict(list)
  ublox_gnss = messaging.sub_sock("ubloxGnss")
  ublox_raw = messaging.sub_sock("ubloxRaw")

  while True:
    num_sv = drain_raw_sock(ublox_raw, raw_ephem, orb_dbs)
    drain_gnss_sock(ublox_gnss, sat_meas, locktime_per_sat, ephem_data)

    if stop_at_fix and num_sv != 0:
      # stop process time for summary
      update_stats(ephem_data, ephem_stats, diff_ephems_per_min, total_ephems_per_min)
      meas_cnt += 1
      print_stats(start_time, ephem_stats, sat_meas, meas_cnt, locktime_per_sat, raw_ephem)
      #print(f"Total     e/min: {sum(total_ephems_per_min)/meas_cnt}")
      #print(f"Different e/min: {sum(diff_ephems_per_min)/meas_cnt}")
      log_test_summary(num_sv, orb_dbs[-1], ephem_data)
      break


    ###########################################################################
    if (int(time.monotonic() - last_log) % 10) == 0:
      if not stop_at_fix:
        print_data(ephem_data)

    if (time.monotonic() - last_log) > 60:
      meas_cnt += 1
      update_stats(ephem_data, ephem_stats, diff_ephems_per_min, total_ephems_per_min)
      if not stop_at_fix:
        print_stats(start_time, ephem_stats, sat_meas, meas_cnt, locktime_per_sat, raw_ephem)
        #print(f"Total     e/min: {sum(total_ephems_per_min)/meas_cnt}")
        #print(f"Different e/min: {sum(diff_ephems_per_min)/meas_cnt}")

      # clear minute data
      #ephem_data = defaultdict(int)
      sat_meas = defaultdict(int)
      locktime_per_sat = {}
      last_log = time.monotonic()


def main(run_test=False):
  print(f"RUN TEST: {run_test}")

  if not run_test:
    # runs endless as listener
    log_loop()

  # test mode
  while True:
    procs = ['pigeond', 'ubloxd']
    for p in procs:
      managed_processes[p].start()
    time.sleep(5)

    print("************************************ START TESTRUN")
    log_loop(True)

    for p in procs:
      managed_processes[p].stop()
    break


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Log ublox output in all the greatness.")
  parser.add_argument("-t", "--run_test", action='store_true', default=False, help='Start processes by script')
  args = parser.parse_args()
  main(args.run_test)
