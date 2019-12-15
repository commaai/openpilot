import unittest
import numpy as np

from cereal import log
import selfdrive.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.planner import calc_cruise_accel_limits
from selfdrive.controls.lib.speed_smoother import speed_smoother
from selfdrive.controls.lib.long_mpc import LongitudinalMpc


def RW(v_ego, v_l):
  TR = 1.8
  G = 9.81
  return (v_ego * TR - (v_l - v_ego) * TR + v_ego * v_ego / (2 * G) - v_l * v_l / (2 * G))


class FakePubMaster():
  def send(self, s, data):
    assert data


def run_following_distance_simulation(v_lead, t_end=200.0):
  dt = 0.2
  t = 0.

  x_lead = 200.0

  x_ego = 0.0
  v_ego = v_lead
  a_ego = 0.0

  v_cruise_setpoint = v_lead + 10.

  pm = FakePubMaster()
  mpc = LongitudinalMpc(1)

  first = True
  while t < t_end:
    # Run cruise control
    accel_limits = [float(x) for x in calc_cruise_accel_limits(v_ego)]
    jerk_limits = [min(-0.1, accel_limits[0]), max(0.1, accel_limits[1])]
    v_cruise, a_cruise = speed_smoother(v_ego, a_ego, v_cruise_setpoint,
                                        accel_limits[1], accel_limits[0],
                                        jerk_limits[1], jerk_limits[0],
                                        dt)

    # Setup CarState
    CS = messaging.new_message()
    CS.init('carState')
    CS.carState.vEgo = v_ego
    CS.carState.aEgo = a_ego

    # Setup lead packet
    lead = log.RadarState.LeadData.new_message()
    lead.status = True
    lead.dRel = x_lead - x_ego
    lead.vLead = v_lead
    lead.aLeadK = 0.0

    # Run MPC
    mpc.set_cur_state(v_ego, a_ego)
    if first:  # Make sure MPC is converged on first timestep
      for _ in range(20):
        mpc.update(pm, CS.carState, lead, v_cruise_setpoint)
    mpc.update(pm, CS.carState, lead, v_cruise_setpoint)

    # Choose slowest of two solutions
    if v_cruise < mpc.v_mpc:
      v_ego, a_ego = v_cruise, a_cruise
    else:
      v_ego, a_ego = mpc.v_mpc, mpc.a_mpc

    # Update state
    x_lead += v_lead * dt
    x_ego += v_ego * dt
    t += dt
    first = False

  return x_lead - x_ego


class TestFollowingDistance(unittest.TestCase):
  def test_following_distanc(self):
    for speed_mph in np.linspace(10, 100, num=10):
      v_lead = float(speed_mph * CV.MPH_TO_MS)

      simulation_steady_state = run_following_distance_simulation(v_lead)
      correct_steady_state = RW(v_lead, v_lead) + 4.0

      self.assertAlmostEqual(simulation_steady_state, correct_steady_state, delta=0.1)
