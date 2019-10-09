#!/usr/bin/env python3
import os
import sys
import argparse

from cereal import log
from common.basedir import BASEDIR
os.environ['BASEDIR'] = BASEDIR


def get_arg_parser():
  parser = argparse.ArgumentParser(
      description="Compare two result files",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("dir1", nargs='?', default='/data/ubloxdc',
                      help="Directory path 1 from which events are loaded")

  parser.add_argument("dir2", nargs='?', default='/data/ubloxdpy',
                      help="Directory path 2 from which msgs are loaded")

  return parser


def read_file(fn):
  with open(fn, 'rb') as f:
    return f.read()


def compare_results(dir1, dir2):
  onlyfiles1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
  onlyfiles1.sort()

  onlyfiles2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]
  onlyfiles2.sort()

  if len(onlyfiles1) != len(onlyfiles2):
    print('len mismatch: {} != {}'.format(len(onlyfiles1), len(onlyfiles2)))
    return -1
  events1 = [log.Event.from_bytes(read_file(os.path.join(dir1, f))) for f in onlyfiles1]
  events2 = [log.Event.from_bytes(read_file(os.path.join(dir2, f))) for f in onlyfiles2]

  for i in range(len(events1)):
    if events1[i].which() != events2[i].which():
      print('event {} type mismatch: {} != {}'.format(i, events1[i].which(), events2[i].which()))
      return -2
    if events1[i].which() == 'gpsLocationExternal':
      old_gps = events1[i].gpsLocationExternal
      gps = events2[i].gpsLocationExternal
      # print(gps, old_gps)
      attrs = ['flags', 'latitude', 'longitude', 'altitude', 'speed', 'bearing',
               'accuracy', 'timestamp', 'source', 'vNED', 'verticalAccuracy', 'bearingAccuracy', 'speedAccuracy']
      for attr in attrs:
        o = getattr(old_gps, attr)
        n = getattr(gps, attr)
        if attr == 'vNED':
          if len(o) != len(n):
            print('Gps vNED len mismatch', o, n)
            return -3
          else:
            for i in range(len(o)):
              if abs(o[i] - n[i]) > 1e-3:
                print('Gps vNED mismatch', o, n)
                return
        elif o != n:
          print('Gps mismatch', attr, o, n)
          return -4
    elif events1[i].which() == 'ubloxGnss':
      old_gnss = events1[i].ubloxGnss
      gnss = events2[i].ubloxGnss
      if old_gnss.which() == 'measurementReport' and gnss.which() == 'measurementReport':
        attrs = ['gpsWeek', 'leapSeconds', 'measurements', 'numMeas', 'rcvTow', 'receiverStatus', 'schema']
        for attr in attrs:
          o = getattr(old_gnss.measurementReport, attr)
          n = getattr(gnss.measurementReport, attr)
          if str(o) != str(n):
            print('measurementReport {} mismatched'.format(attr))
            return -5
        if not (str(old_gnss.measurementReport) == str(gnss.measurementReport)):
          print('Gnss measurementReport mismatched!')
          print('gnss measurementReport old', old_gnss.measurementReport.measurements)
          print('gnss measurementReport new', gnss.measurementReport.measurements)
          return -6
      elif old_gnss.which() == 'ephemeris' and gnss.which() == 'ephemeris':
        if not (str(old_gnss.ephemeris) == str(gnss.ephemeris)):
          print('Gnss ephemeris mismatched!')
          print('gnss ephemeris old', old_gnss.ephemeris)
          print('gnss ephemeris new', gnss.ephemeris)
          return -7
  print('All {} events matched!'.format(len(events1)))
  return 0


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  compare_results(args.dir1, args.dir2)
