import os
import sys
import math
import platform
import numpy as np

from common.numpy_fast import clip, interp
from common.kalman.ekf import FastEKF1D, SimpleSensor

# radar tracks
SPEED, ACCEL = 0, 1   # Kalman filter states enum

rate, ratev = 20., 20.    # model and radar are both at 20Hz
ts = 1./rate
freq_v_lat = 0.2 # Hz
k_v_lat = 2*np.pi*freq_v_lat*ts / (1 + 2*np.pi*freq_v_lat*ts)

freq_a_lead = .5 # Hz
k_a_lead = 2*np.pi*freq_a_lead*ts / (1 + 2*np.pi*freq_a_lead*ts)

# stationary qualification parameters
v_stationary_thr = 4.   # objects moving below this speed are classified as stationary
v_oncoming_thr   = -3.9 # needs to be a bit lower in abs value than v_stationary_thr to not leave "holes"
v_ego_stationary = 4.   # no stationary object flag below this speed

class Track(object):
  def __init__(self):
    self.ekf = None
    self.stationary = True
    self.initted = False

  def update(self, d_rel, y_rel, v_rel, d_path, v_ego_t_aligned):
    if self.initted:
      self.dPathPrev = self.dPath
      self.vLeadPrev = self.vLead
      self.vRelPrev = self.vRel

    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED

    # compute distance to path
    self.dPath = d_path

    # computed velocity and accelerations
    self.vLead = self.vRel + v_ego_t_aligned

    if not self.initted:
      self.aRel = 0.      # nidec gives no information about this
      self.vLat = 0.
      self.aLead = 0.
    else:
      # estimate acceleration
      a_rel_unfilt = (self.vRel - self.vRelPrev) / ts
      a_rel_unfilt = clip(a_rel_unfilt, -10., 10.)
      self.aRel = k_a_lead * a_rel_unfilt + (1 - k_a_lead) * self.aRel

      v_lat_unfilt = (self.dPath - self.dPathPrev) / ts
      self.vLat = k_v_lat * v_lat_unfilt + (1 - k_v_lat) * self.vLat

      a_lead_unfilt = (self.vLead - self.vLeadPrev) / ts
      a_lead_unfilt = clip(a_lead_unfilt, -10., 10.)
      self.aLead = k_a_lead * a_lead_unfilt + (1 - k_a_lead) * self.aLead

    if self.stationary:
      # stationary objects can become non stationary, but not the other way around
      self.stationary = v_ego_t_aligned > v_ego_stationary and abs(self.vLead) < v_stationary_thr
    self.oncoming = self.vLead < v_oncoming_thr

    if self.ekf is None:
      self.ekf = FastEKF1D(ts, 1e3, [0.1, 1])
      self.ekf.state[SPEED] = self.vLead
      self.ekf.state[ACCEL] = 0
      self.lead_sensor = SimpleSensor(SPEED, 1, 2)

      self.vLeadK = self.vLead
      self.aLeadK = self.aLead
    else:
      self.ekf.update_scalar(self.lead_sensor.read(self.vLead))
      self.ekf.predict(ts)
      self.vLeadK = float(self.ekf.state[SPEED])
      self.aLeadK = float(self.ekf.state[ACCEL])

    if not self.initted:
      self.cnt = 1
      self.vision_cnt = 0
    else:
      self.cnt += 1

    self.initted = True
    self.vision = False

  def mix_vision(self, dist_to_vision, rel_speed_diff):
    # rel speed is very hard to estimate from vision
    if dist_to_vision < 4.0 and rel_speed_diff < 10.:
      # vision point is never stationary
      self.stationary = False
      self.vision = True
      self.vision_cnt += 1

  def get_key_for_cluster(self):
    # Weigh y higher since radar is inaccurate in this dimension
    return [self.dRel, self.dPath*2, self.vRel]

# ******************* Cluster *******************

if platform.machine() == 'aarch64':
  for x in sys.path:
    pp = os.path.join(x, "phonelibs/hierarchy/lib")
    if os.path.isfile(os.path.join(pp, "_hierarchy.so")):
      sys.path.append(pp)
      break
  import _hierarchy
else:
  from scipy.cluster import _hierarchy

def fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
  # supersimplified function to get fast clustering. Got it from scipy
  Z = np.asarray(Z, order='c')
  n = Z.shape[0] + 1
  T = np.zeros((n,), dtype='i')
  _hierarchy.cluster_dist(Z, T, float(t), int(n))
  return T

RDR_TO_LDR = 2.7

def mean(l):
  return sum(l)/len(l)

class Cluster(object):
  def __init__(self):
    self.tracks = set()

  def add(self, t):
    # add the first track
    self.tracks.add(t)

  # TODO: make generic
  @property
  def dRel(self):
    return mean([t.dRel for t in self.tracks])

  @property
  def yRel(self):
    return mean([t.yRel for t in self.tracks])

  @property
  def vRel(self):
    return mean([t.vRel for t in self.tracks])

  @property
  def aRel(self):
    return mean([t.aRel for t in self.tracks])

  @property
  def vLead(self):
    return mean([t.vLead for t in self.tracks])

  @property
  def aLead(self):
    return mean([t.aLead for t in self.tracks])

  @property
  def dPath(self):
    return mean([t.dPath for t in self.tracks])

  @property
  def vLat(self):
    return mean([t.vLat for t in self.tracks])

  @property
  def vLeadK(self):
    return mean([t.vLeadK for t in self.tracks])

  @property
  def aLeadK(self):
    return mean([t.aLeadK for t in self.tracks])

  @property
  def vision(self):
    return any([t.vision for t in self.tracks])

  @property
  def vision_cnt(self):
    return max([t.vision_cnt for t in self.tracks])

  @property
  def stationary(self):
    return all([t.stationary for t in self.tracks])

  @property
  def oncoming(self):
    return all([t.oncoming for t in self.tracks])

  def toLive20(self, lead):
    lead.dRel = float(self.dRel) - RDR_TO_LDR
    lead.yRel = float(self.yRel)
    lead.vRel = float(self.vRel)
    lead.aRel = float(self.aRel)
    lead.vLead = float(self.vLead)
    lead.aLead = float(self.aLead)
    lead.dPath = float(self.dPath)
    lead.vLat = float(self.vLat)
    lead.vLeadK = float(self.vLeadK)
    lead.aLeadK = float(self.aLeadK)
    lead.status = True
    lead.fcw = False

  def __str__(self):
    ret = "x: %7.2f  y: %7.2f  v: %7.2f  a: %7.2f" % (self.dRel, self.yRel, self.vRel, self.aRel)
    if self.stationary:
      ret += " stationary"
    if self.vision:
      ret += " vision"
    if self.oncoming:
      ret += " oncoming"
    if self.vision_cnt > 0:
      ret += " vision_cnt: %6.0f" % self.vision_cnt
    return ret

  def is_potential_lead(self, v_ego):
    # predict cut-ins by extrapolating lateral speed by a lookahead time
    # lookahead time depends on cut-in distance. more attentive for close cut-ins
    # also, above 50 meters the predicted path isn't very reliable

    # the distance at which v_lat matters is higher at higher speed
    lookahead_dist = 40. + v_ego/1.2   #40m at 0mph, ~70m at 80mph

    t_lookahead_v  = [1., 0.]
    t_lookahead_bp = [10., lookahead_dist]

    # average dist
    d_path = self.dPath

    # lat_corr used to be gated on enabled, now always running
    t_lookahead = interp(self.dRel, t_lookahead_bp, t_lookahead_v)
    # correct d_path for lookahead time, considering only cut-ins and no more than 1m impact
    lat_corr = clip(t_lookahead * self.vLat, -1, 0)

    d_path = max(d_path + lat_corr, 0)

    if d_path < 1.5 and not self.stationary and not self.oncoming:
      return True
    else:
      return False

  def is_potential_lead2(self, lead_clusters):
    if len(lead_clusters) > 0:
      lead_cluster = lead_clusters[0]
      # check if the new lead is too close and roughly at the same speed of the first lead: it might just be the second axle of the same vehicle
      if (self.dRel - lead_cluster.dRel) < 8. and abs(self.vRel - lead_cluster.vRel) < 1.:
        return False
      else:
        return True
    else:
      return False
