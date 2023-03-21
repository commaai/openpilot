#!/usr/bin/env python

# Copyright (C) 2014 Swift Navigation Inc.
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

import datetime
import numpy as np
import logging

def floatornan(x):
  if x == '' or x[-1] == ' ':
    return np.NaN
  return float(x)


def digitorzero(x):
  if x == ' ' or x == '':
    return 0
  return int(x)


def padline(l, n=16):
  x = len(l)
  x_ = n * ((x + n - 1) // n)
  padded = l + ' ' * (x_ - x)
  while len(padded) < 70:
    padded += ' ' * 16
  return padded


TOTAL_SATS = 132  # Increased to support Galileo


class DownloadError(Exception):
  pass


class RINEXFile:
  def __init__(self, filename, rate=None):
    self.rate = rate
    try:
      with open(filename) as f:
        self._read_header(f)
        self._read_data(f)
    except TypeError:
      logging.exception("TypeError, file likely not downloaded.")
      raise DownloadError("file download failure")
    except FileNotFoundError:
      logging.exception("File not found in directory.")
      raise DownloadError("file missing in download cache")
  def _read_header(self, f):
    version_line = padline(f.readline(), 80)

    self.version = float(version_line[0:9])
    if (self.version > 2.11):
      raise ValueError(
        f"RINEX file versions > 2.11 not supported (file version {self.version:f})")

    self.filetype = version_line[20]
    if self.filetype not in "ONGM":  # Check valid file type
      raise ValueError(f"RINEX file type '{self.filetype}' not supported")
    if self.filetype != 'O':
      raise ValueError("Only 'OBSERVATION DATA' RINEX files are currently supported")

    self.gnss = version_line[40]
    if self.gnss not in " GRSEM":  # Check valid satellite system
      raise ValueError(f"Satellite system '{self.filetype}' not supported")
    if self.gnss == ' ':
      self.gnss = 'G'
    if self.gnss != 'G':
      #raise ValueError("Only GPS data currently supported")
      pass

    self.comment = ""
    while True:  # Read the rest of the header
      line = padline(f.readline(), 80)
      label = line[60:80].rstrip()
      if label == "END OF HEADER":
        break
      if label == "COMMENT":
        self.comment += line[:60] + '\n'
      if label == "MARKER NAME":
        self.marker_name = line[:60].rstrip()
        if self.marker_name == '':
          self.marker_name = 'UNKNOWN'
      if label == "# / TYPES OF OBSERV":
        # RINEX files can have multiple line headers
        # This code handles the case
        try:
          n_obs = int(line[0:6])
          self.obs_types = []
        except ValueError:
          pass

        if n_obs <= 9:
          for i in range(0, n_obs):
            self.obs_types.append(line[10 + 6 * i:12 + 6 * i])
        if n_obs > 9:
          for i in range(0, 9):
            self.obs_types.append(line[10 + 6 * i:12 + 6 * i])
          n_obs -= 9

  def _read_next_non_comment(self, f):
    line = f.readline()
    while line and line.find('COMMENT') != -1:
      line = f.readline()
    return line

  def _read_epoch_header(self, f):
    epoch_hdr = self._read_next_non_comment(f)
    if epoch_hdr == '':
      return None
    # ignore any line with these three strings
    skippable = ('0.0000000  4  5', 'MARKER NUMBER', '          4  1')
    while any(skip in epoch_hdr for skip in skippable):
      epoch_hdr = self._read_next_non_comment(f)

    if epoch_hdr == '':
      return None

    year = int(epoch_hdr[1:3])
    if year >= 80:
      year += 1900
    else:
      year += 2000
    month = int(epoch_hdr[4:6])
    day = int(epoch_hdr[7:9])
    hour = int(epoch_hdr[10:12])
    minute = int(epoch_hdr[13:15])
    second = int(epoch_hdr[15:18])
    microsecond = int(
      epoch_hdr[19:25])  # Discard the least sig. fig. (use microseconds only).
    epoch = datetime.datetime(year, month, day, hour, minute, second, microsecond)

    flag = int(epoch_hdr[28])
    allowed_flags = {0, 3, 4}
    if flag not in allowed_flags:
      raise ValueError("Don't know how to handle epoch flag %d in epoch header:\n%s" %
                       (flag, epoch_hdr))

    n_sats = int(epoch_hdr[29:32])
    if flag > 1:  # event flag: nsats is number of records
      for i in range(n_sats):
        f.readline()
      return None

    sats = []
    for i in range(0, n_sats):
      if ((i % 12) == 0) and (i > 0):
        epoch_hdr = f.readline()
      sats.append(epoch_hdr[(32 + (i % 12) * 3):(35 + (i % 12) * 3)])

    return epoch, flag, sats

  def _read_obs(self, f, n_sat, sat_map):
    obs = np.empty((TOTAL_SATS, len(self.obs_types)), dtype=np.float64) * np.NaN
    lli = np.zeros((TOTAL_SATS, len(self.obs_types)), dtype=np.uint8)
    signal_strength = np.zeros((TOTAL_SATS, len(self.obs_types)), dtype=np.uint8)

    for i in range(n_sat):
      # Join together observations for a single satellite if split across lines.
      obs_line = ''.join(
        padline(f.readline()[:-1], 16) for _ in range((len(self.obs_types) + 4) // 5))
      for j in range(len(self.obs_types)):
        obs_record = obs_line[16 * j:16 * (j + 1)]
        obs[int(sat_map[i]), j] = floatornan(obs_record[0:14])
        lli[int(sat_map[i]), j] = digitorzero(obs_record[14:15])
        signal_strength[int(sat_map[i]), j] = digitorzero(obs_record[15:16])

    return obs, lli, signal_strength

  def _skip_obs(self, f, n_sat):
    for i in range(n_sat):
      for _ in range((len(self.obs_types) + 4) // 5):
        f.readline()

  def _read_data_chunk(self, f, CHUNK_SIZE=10000):
    obss = np.empty(
      (CHUNK_SIZE, TOTAL_SATS, len(self.obs_types)), dtype=np.float64) * np.NaN
    llis = np.zeros((CHUNK_SIZE, TOTAL_SATS, len(self.obs_types)), dtype=np.uint8)
    signal_strengths = np.zeros(
      (CHUNK_SIZE, TOTAL_SATS, len(self.obs_types)), dtype=np.uint8)
    epochs = np.zeros(CHUNK_SIZE, dtype='datetime64[us]')
    flags = np.zeros(CHUNK_SIZE, dtype=np.uint8)

    i = 0
    while True:
      hdr = self._read_epoch_header(f)
      if hdr is None:
        break
      # data faster than desired rate: ignore it
      if self.rate and (hdr[0].microsecond or hdr[0].second % self.rate != 0):
        self._skip_obs(f, len(hdr[2]))
        continue
      epoch, flags[i], sats = hdr
      epochs[i] = np.datetime64(epoch)
      sat_map = np.ones(len(sats)) * -1
      for n, sat in enumerate(sats):
        if sat[0] == 'G':
          sat_map[n] = int(sat[1:]) - 1
        if sat[0] == 'R':
          sat_map[n] = int(sat[1:]) - 1 + 64
      obss[i], llis[i], signal_strengths[i] = self._read_obs(f, len(sats), sat_map)
      i += 1
      if i >= CHUNK_SIZE:
        break

    return obss[:i], llis[:i], signal_strengths[:i], epochs[:i], flags[:i]

  def _read_data(self, f):
    self.data = {}
    while True:
      obss, llis, signal_strengths, epochs, flags = self._read_data_chunk(f)
      if obss.shape[0] == 0:
        break

      for i, sv in enumerate(['%02d' % d for d in range(1, TOTAL_SATS+1)]):
        if sv not in self.data:
          self.data[sv] = {}
        for j, obs_type in enumerate(self.obs_types):
          if obs_type in self.data[sv]:
            self.data[sv][obs_type] = np.append(self.data[sv][obs_type], obss[:, i, j])
          else:
            self.data[sv][obs_type] = obss[:, i, j]
        if 'Epochs' in self.data[sv]:
          self.data[sv]['Epochs'] = np.append(self.data[sv]['Epochs'], epochs)
        else:
          self.data[sv]['Epochs'] = epochs
    for sat in list(self.data.keys()):
      if np.all(np.isnan(self.data[sat]['C1'])):
        del self.data[sat]















