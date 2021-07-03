#!/usr/bin/env python3
import unittest
import numpy as np

from cereal import log
import cereal.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.lead_mpc import LeadMpc


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

  mpc = LeadMpc(0)

  first = True
  while t < t_end:

    # Setup CarState
    CS = messaging.new_message('carState')
    CS.carState.vEgo = v_ego
    CS.carState.aEgo = a_ego

    # Setup model packet
    radarstate = messaging.new_message('radarState')
    lead = log.RadarState.LeadData.new_message()
    lead.modelProb = .75
    lead.dRel = x_lead - x_ego
    lead.vLead = v_lead
    lead.aLeadK = 0.0
    lead.status = True
    radarstate.radarState.leadOne = lead

    # Run MPC
    mpc.set_cur_state(v_ego, a_ego)
    if first:  # Make sure MPC is converged on first timestep
      for _ in range(20):
        mpc.update(CS.carState, radarstate.radarState, 0)
    mpc.update(CS.carState, radarstate.radarState, 0)

    # Choose slowest of two solutions
    v_ego, a_ego = mpc.mpc_solution.v_ego[5], mpc.mpc_solution.a_ego[5]

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

      self.assertAlmostEqual(simulation_steady_state, correct_steady_state, delta=.1)


if __name__ == "__main__":
  unittest.main()
