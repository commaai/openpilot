import unittest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import KalmanParams, Track, cluster_track_accels


class TestRadarTrack(unittest.TestCase):
  def test_cluster_acceleration_stays_stable_when_track_switches(self):
    kalman_params = KalmanParams(DT_MDL)

    first_track = Track(0, 20.0, kalman_params)
    first_track.update(30.0, 0.0, 0.0, 20.0)
    first_track.update(30.0, 0.0, 0.0, 20.0)
    first_track.set_accel(-2.0, 0.8)

    second_track = Track(1, 20.2, kalman_params)
    second_track.update(30.2, 0.1, 0.2, 20.2)
    second_track.update(30.2, 0.1, 0.2, 20.2)
    second_track.set_accel(-1.0, 1.0)

    replacement_track = Track(2, 20.1, kalman_params)
    replacement_track.update(30.1, 0.05, 0.1, 20.1)

    unrelated_track = Track(3, 20.0, kalman_params)
    unrelated_track.update(40.0, 0.0, 0.0, 20.0)

    self.assertEqual(replacement_track.aLeadK, 0.0)
    cluster_accels = cluster_track_accels({
      track.identifier: track for track in (first_track, second_track, replacement_track, unrelated_track)
    })

    for track in (first_track, second_track, replacement_track):
      a_lead, a_lead_tau = cluster_accels[track.identifier]
      self.assertAlmostEqual(a_lead, -1.5)
      self.assertAlmostEqual(a_lead_tau, 0.9)

    first_lead = first_track.get_RadarState(accel=cluster_accels[first_track.identifier])
    replacement_lead = replacement_track.get_RadarState(accel=cluster_accels[replacement_track.identifier])
    self.assertAlmostEqual(first_lead["aLeadK"], replacement_lead["aLeadK"])

    self.assertAlmostEqual(replacement_track.aLeadK, -1.5)
    self.assertAlmostEqual(replacement_track.aLeadTau.x, 0.9)
    self.assertAlmostEqual(unrelated_track.aLeadK, 0.0)
    self.assertEqual(cluster_accels[unrelated_track.identifier], (0.0, 1.5))

  def test_cluster_threshold(self):
    kalman_params = KalmanParams(DT_MDL)

    mature_track = Track(0, 20.0, kalman_params)
    mature_track.update(30.0, 0.0, 0.0, 20.0)
    mature_track.update(30.0, 0.0, 0.0, 20.0)
    mature_track.set_accel(-2.0, 0.8)

    inside_track = Track(1, 20.0, kalman_params)
    inside_track.update(32.49, 0.0, 0.0, 20.0)
    boundary_track = Track(2, 20.0, kalman_params)
    boundary_track.update(32.5, 0.0, 0.0, 20.0)

    cluster_accels = cluster_track_accels({
      track.identifier: track for track in (mature_track, inside_track, boundary_track)
    })

    self.assertEqual(cluster_accels[inside_track.identifier], (-2.0, 0.8))
    self.assertEqual(cluster_accels[boundary_track.identifier], (0.0, 1.5))

  def test_inherited_acceleration_survives_next_update(self):
    kalman_params = KalmanParams(DT_MDL)

    mature_track = Track(0, 20.0, kalman_params)
    mature_track.update(30.0, 0.0, 0.0, 20.0)
    mature_track.update(30.0, 0.0, 0.0, 20.0)
    mature_track.set_accel(-2.0, 0.8)

    replacement_track = Track(1, 20.0, kalman_params)
    replacement_track.update(30.1, 0.0, 0.0, 20.0)
    cluster_track_accels({track.identifier: track for track in (mature_track, replacement_track)})

    replacement_track.update(30.1, 0.0, -0.1, 19.9)
    self.assertLess(replacement_track.aLeadK, -1.5)
