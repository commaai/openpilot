#!/usr/bin/env python
from common.kalman.ned import ecef2geodetic

import csv
import numpy as np
import webbrowser
import os
import sys
import json
import numpy.linalg as LA
import gmplot
from dateutil.parser import parse
from common.numpy_helpers import deep_interp
# import cvxpy as cvx
MPH_TO_MS = 0.44704


def downsample(positions, speeds, start_idx, end_idx, dist):
    # TODO: save headings too
    track = []
    last_position = positions[start_idx]
    valid_indeces = []
    track_speeds = []
    for pi in range(start_idx, end_idx):
        # only save points that are at least 10 cm far away
        if LA.norm(positions[pi] - last_position) >= dist:
            #print LA.norm(positions[pi] - last_position)
            last_position = positions[pi]
            track.append(positions[pi])
            valid_indeces.append(pi)
            track_speeds.append(speeds[pi])
    print(-start_idx + end_idx, len(valid_indeces))
    # this compare the original point count vs the filtered count 
            
    track = np.array(track)
    track_speeds = np.array(track_speeds)
    return track, track_speeds

def converter(date):

  filename = "/home/batman/one/selfdrive/locationd/liveloc_dumps/" + date + "/canonical.csv"  # Point one (OK!)
  
  c = csv.DictReader(open(filename, 'rb'), delimiter=',')
  
  start_time = None
  
  t = []
  ll_positions = []
  positions = []
  sats = []
  flag = []
  speeds = []
  
  for row in c:
      t.append(float(row['pctime']))
      x = float(row['ecefX'])
      y = float(row['ecefY'])
      z = float(row['ecefZ'])
      ecef = np.array((x, y, z))
      speeds.append(float(row['velSpeed']))
                      
      pos = ecef2geodetic(ecef)
      ll_positions.append(pos)
      positions.append(ecef)
  
  t = np.array(t)
  ll_positions = np.array(ll_positions)
  positions = np.array(positions)
                      
  #distances = ll_positions[:,0:2] - START_POS[:2]
  #i_start = np.argmin(LA.norm(distances, axis=1))
  
  #for i in range(i_start + 500):
  #    distances[i] += np.array([100, 100])
  #i_end = np.argmin(LA.norm(distances, axis=1))

  i_start = 0
  i_end = len(positions)
  
  print(i_start, i_end)
  track, track_speeds = downsample(positions, speeds, i_start, i_end, 0.2)
  ll_track = np.array([ecef2geodetic(pos) for pos in track])
  
  track_struct = {}
  print(track_speeds.shape)
  print(track.shape)
  track_struct['race'] = np.hstack((track, 
                                    np.expand_dims(track_speeds, axis=1), 
                                    np.zeros((len(track_speeds), 1))))
  
  f = open('/home/batman/one/selfdrive/controls/tracks/loop_city.npy', 'w')
  np.save(f, track_struct)
  f.close()
  print("SAVED!")


if __name__ == "__main__":
  converter(sys.argv[1])
