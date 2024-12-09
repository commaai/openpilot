import numpy as np
from typing import cast
from collections import defaultdict
from math import cos, sin
from dataclasses import dataclass
from opendbc.can.parser import CANParser
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.ford.fordcan import CanBus
from opendbc.car.ford.values import DBC, RADAR
from opendbc.car.interfaces import RadarInterfaceBase

DELPHI_ESR_RADAR_MSGS = list(range(0x500, 0x540))

DELPHI_MRR_RADAR_START_ADDR = 0x120
DELPHI_MRR_RADAR_HEADER_ADDR = 0x174  # MRR_Header_SensorCoverage
DELPHI_MRR_RADAR_MSG_COUNT = 64

DELPHI_MRR_RADAR_RANGE_COVERAGE = {0: 42, 1: 164, 2: 45, 3: 175}  # scan index to detection range (m)
DELPHI_MRR_MIN_LONG_RANGE_DIST = 30  # meters
DELPHI_MRR_CLUSTER_THRESHOLD = 5  # meters, lateral distance and relative velocity are weighted


@dataclass
class Cluster:
  dRel: float = 0.0
  yRel: float = 0.0
  vRel: float = 0.0
  trackId: int = 0


def cluster_points(pts_l: list[list[float]], pts2_l: list[list[float]], max_dist: float) -> list[int]:
  """
  Clusters a collection of points based on another collection of points. This is useful for correlating clusters through time.
  Points in pts2 not close enough to any point in pts are assigned -1.
  Args:
    pts_l: List of points to base the new clusters on
    pts2_l: List of points to cluster using pts
    max_dist: Max distance from cluster center to candidate point

  Returns:
    List of cluster indices for pts2 that correspond to pts
  """

  if not len(pts2_l):
    return []

  if not len(pts_l):
    return [-1] * len(pts2_l)

  max_dist_sq = max_dist ** 2
  pts = np.array(pts_l)
  pts2 = np.array(pts2_l)

  # Compute squared norms
  pts_norm_sq = np.sum(pts ** 2, axis=1)
  pts2_norm_sq = np.sum(pts2 ** 2, axis=1)

  # Compute squared Euclidean distances using the identity
  # dist_sq[i, j] = ||pts2[i]||^2 + ||pts[j]||^2 - 2 * pts2[i] . pts[j]
  dist_sq = pts2_norm_sq[:, np.newaxis] + pts_norm_sq[np.newaxis, :] - 2 * np.dot(pts2, pts.T)
  dist_sq = np.maximum(dist_sq, 0.0)

  # Find the closest cluster for each point and assign its index
  closest_clusters = np.argmin(dist_sq, axis=1)
  closest_dist_sq = dist_sq[np.arange(len(pts2)), closest_clusters]
  cluster_idxs = np.where(closest_dist_sq < max_dist_sq, closest_clusters, -1)

  return cast(list[int], cluster_idxs.tolist())


def _create_delphi_esr_radar_can_parser(CP) -> CANParser:
  msg_n = len(DELPHI_ESR_RADAR_MSGS)
  messages = list(zip(DELPHI_ESR_RADAR_MSGS, [20] * msg_n, strict=True))

  return CANParser(RADAR.DELPHI_ESR, messages, CanBus(CP).radar)


def _create_delphi_mrr_radar_can_parser(CP) -> CANParser:
  messages = [
    ("MRR_Header_InformationDetections", 33),
    ("MRR_Header_SensorCoverage", 33),
  ]

  for i in range(1, DELPHI_MRR_RADAR_MSG_COUNT + 1):
    msg = f"MRR_Detection_{i:03d}"
    messages += [(msg, 33)]

  return CANParser(RADAR.DELPHI_MRR, messages, CanBus(CP).radar)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)

    self.points: list[list[float]] = []
    self.clusters: list[Cluster] = []

    self.updated_messages = set()
    self.track_id = 0
    self.radar = DBC[CP.carFingerprint].get(Bus.radar)
    if CP.radarUnavailable:
      self.rcp = None
    elif self.radar == RADAR.DELPHI_ESR:
      self.rcp = _create_delphi_esr_radar_can_parser(CP)
      self.trigger_msg = DELPHI_ESR_RADAR_MSGS[-1]
      self.valid_cnt = {key: 0 for key in DELPHI_ESR_RADAR_MSGS}
    elif self.radar == RADAR.DELPHI_MRR:
      self.rcp = _create_delphi_mrr_radar_can_parser(CP)
      self.trigger_msg = DELPHI_MRR_RADAR_HEADER_ADDR
    else:
      raise ValueError(f"Unsupported radar: {self.radar}")

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None
    self.updated_messages.clear()

    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")

    if self.radar == RADAR.DELPHI_ESR:
      self._update_delphi_esr()
    elif self.radar == RADAR.DELPHI_MRR:
      _update, _errors = self._update_delphi_mrr()
      errors.extend(_errors)
      if not _update:
        return None

    ret = structs.RadarData()
    ret.points = list(self.pts.values())
    ret.errors = errors
    return ret

  def _update_delphi_esr(self):
    for ii in sorted(self.updated_messages):
      cpt = self.rcp.vl[ii]

      if cpt['X_Rel'] > 0.00001:
        self.valid_cnt[ii] = 0    # reset counter

      if cpt['X_Rel'] > 0.00001:
        self.valid_cnt[ii] += 1
      else:
        self.valid_cnt[ii] = max(self.valid_cnt[ii] - 1, 0)
      #print ii, self.valid_cnt[ii], cpt['VALID'], cpt['X_Rel'], cpt['Angle']

      # radar point only valid if there have been enough valid measurements
      if self.valid_cnt[ii] > 0:
        if ii not in self.pts:
          self.pts[ii] = structs.RadarData.RadarPoint()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['X_Rel']  # from front of car
        self.pts[ii].yRel = cpt['X_Rel'] * cpt['Angle'] * CV.DEG_TO_RAD  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['V_Rel']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True
      else:
        if ii in self.pts:
          del self.pts[ii]

  def _update_delphi_mrr(self):
    headerScanIndex = int(self.rcp.vl["MRR_Header_InformationDetections"]['CAN_SCAN_INDEX']) & 0b11

    # Use points with Doppler coverage of +-60 m/s, reduces similar points
    if headerScanIndex in (0, 1):
      return False, []

    errors = []
    if DELPHI_MRR_RADAR_RANGE_COVERAGE[headerScanIndex] != int(self.rcp.vl["MRR_Header_SensorCoverage"]["CAN_RANGE_COVERAGE"]):
      errors.append("wrongConfig")

    for ii in range(1, DELPHI_MRR_RADAR_MSG_COUNT + 1):
      msg = self.rcp.vl[f"MRR_Detection_{ii:03d}"]

      # SCAN_INDEX rotates through 0..3 on each message for different measurement modes
      # Indexes 0 and 2 have a max range of ~40m, 1 and 3 are ~170m (MRR_Header_SensorCoverage->CAN_RANGE_COVERAGE)
      # Indexes 0 and 1 have a Doppler coverage of +-71 m/s, 2 and 3 have +-60 m/s
      scanIndex = msg[f"CAN_SCAN_INDEX_2LSB_{ii:02d}"]

      # Throw out old measurements. Very unlikely to happen, but is proper behavior
      if scanIndex != headerScanIndex:
        continue

      valid = bool(msg[f"CAN_DET_VALID_LEVEL_{ii:02d}"])

      # Long range measurement mode is more sensitive and can detect the road surface
      dist = msg[f"CAN_DET_RANGE_{ii:02d}"]  # m [0|255.984]
      if scanIndex in (1, 3) and dist < DELPHI_MRR_MIN_LONG_RANGE_DIST:
        valid = False

      if valid:
        azimuth = msg[f"CAN_DET_AZIMUTH_{ii:02d}"]              # rad [-3.1416|3.13964]
        distRate = msg[f"CAN_DET_RANGE_RATE_{ii:02d}"]          # m/s [-128|127.984]
        dRel = cos(azimuth) * dist                              # m from front of car
        yRel = -sin(azimuth) * dist                             # in car frame's y axis, left is positive

        self.points.append([dRel, yRel * 2, distRate * 2])

    # Update once we've cycled through all 4 scan modes
    if headerScanIndex != 3:
      return False, []

    # Cluster points from this cycle against the centroids from the previous cycle
    prev_keys = [[p.dRel, p.yRel * 2, p.vRel * 2] for p in self.clusters]
    labels = cluster_points(prev_keys, self.points, DELPHI_MRR_CLUSTER_THRESHOLD)

    points_by_track_id = defaultdict(list)
    for idx, label in enumerate(labels):
      if label != -1:
        points_by_track_id[self.clusters[label].trackId].append(self.points[idx])
      else:
        points_by_track_id[self.track_id].append(self.points[idx])
        self.track_id += 1

    self.clusters = []
    for idx, (track_id, pts) in enumerate(points_by_track_id.items()):
      dRel = [p[0] for p in pts]
      min_dRel = min(dRel)
      dRel = sum(dRel) / len(dRel)

      yRel = [p[1] for p in pts]
      yRel = sum(yRel) / len(yRel) / 2

      vRel = [p[2] for p in pts]
      vRel = sum(vRel) / len(vRel) / 2

      # FIXME: creating capnp RadarPoint and accessing attributes are both expensive, so we store a dataclass and reuse the RadarPoint
      self.clusters.append(Cluster(dRel=dRel, yRel=yRel, vRel=vRel, trackId=track_id))

      if idx not in self.pts:
        self.pts[idx] = structs.RadarData.RadarPoint(measured=True, aRel=float('nan'), yvRel=float('nan'))

      self.pts[idx].dRel = min_dRel
      self.pts[idx].yRel = yRel
      self.pts[idx].vRel = vRel
      self.pts[idx].trackId = track_id

    for idx in range(len(points_by_track_id), len(self.pts)):
      del self.pts[idx]

    self.points = []

    return True, errors
