import unittest
import json
import numpy as np

from cereal import log
import cereal.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.longitudinal_planner import calc_cruise_accel_limits
from selfdrive.controls.lib.speed_smoother import speed_smoother
from selfdrive.controls.lib.long_mpc import LongitudinalMpc


def RW(v_ego, v_l):
  TR = 1.8
  G = 9.81
  return (v_ego * TR - (v_l - v_ego) * TR + v_ego * v_ego / (2 * G) - v_l * v_l / (2 * G))


class FakePubMaster():
  def send(self, s, data):
    assert data


class LeadPos:
  INITIAL = 0
  IS_BREAKING = 1
  IS_RECOVERING = 2
  FINAL = 3

  def __init__(self, v_target):
    self.v = self.v_target = v_target
    self.x = 200.0
    self.phase = self.INITIAL
  
  def update(self, t, dt):
    self.x += self.v * dt
    
    if t > 150.0:
      if self.phase == self.INITIAL:
        self.phase = self.IS_BREAKING
      elif self.phase == self.IS_BREAKING and self.v <= self.v_target * 0.5:
        self.phase = self.IS_RECOVERING
      elif self.phase == self.IS_RECOVERING and self.v >= self.v_target:
        self.phase = self.FINAL
        self.v = self.v_target
    
    if self.phase == self.IS_BREAKING:
      self.v -= 9.86 / 2 * dt
    elif self.phase == self.IS_RECOVERING:
      self.v += 4. * dt


def run_following_distance_simulation(TR_override, v_lead, t_end=200.0):
  dt = 0.2
  t = 0.

  lead_pos = LeadPos(v_lead)

  x_ego = 0.0
  v_ego = v_lead
  a_ego = 0.0

  v_cruise_setpoint = v_lead + 10.

  pm = FakePubMaster()
  mpc = LongitudinalMpc(1, TR_override)

  datapoints = [{'t': t, 'x_ego': x_ego, 'x_lead': lead_pos.x}]

  first = True
  while t < t_end:
    # Run cruise control
    accel_limits = [float(x) for x in calc_cruise_accel_limits(v_ego, False)]
    jerk_limits = [min(-0.1, accel_limits[0]), max(0.1, accel_limits[1])]
    v_cruise, a_cruise = speed_smoother(v_ego, a_ego, v_cruise_setpoint,
                                        accel_limits[1], accel_limits[0],
                                        jerk_limits[1], jerk_limits[0],
                                        dt)

    # Setup CarState
    CS = messaging.new_message('carState')
    CS.carState.vEgo = v_ego
    CS.carState.aEgo = a_ego

    # Setup lead packet
    lead = log.RadarState.LeadData.new_message()
    lead.status = True
    lead.dRel = lead_pos.x - x_ego
    lead.vLead = lead_pos.v
    lead.aLeadK = 0.0

    # Run MPC
    mpc.set_cur_state(v_ego, a_ego)
    if first:  # Make sure MPC is converged on first timestep
      for _ in range(20):
        mpc.update(CS.carState, lead)
        mpc.publish(pm)
    mpc.update(CS.carState, lead)
    mpc.publish(pm)

    # Choose slowest of two solutions
    if v_cruise < mpc.v_mpc:
      v_ego, a_ego = v_cruise, a_cruise
    else:
      v_ego, a_ego = mpc.v_mpc, mpc.a_mpc

    # Update state
    lead_pos.update(t, dt)
    x_ego += v_ego * dt
    t += dt
    first = False

    datapoints.append({'t': t, 'x_ego': x_ego, 'x_lead': lead_pos.x})

  filename = f'test_out/{v_lead}_{TR_override}.json'
  with open(filename, 'w') as datafile:
    json.dump(datapoints, datafile)
  return lead_pos.x - x_ego


class TestFollowingDistance(unittest.TestCase):
  def test_following_distanc(self):
    for speed_mph in np.linspace(10, 100, num=10):
      v_lead = float(speed_mph * CV.MPH_TO_MS)

      simulation_steady_state = run_following_distance_simulation(None, v_lead)
      correct_steady_state = RW(v_lead, v_lead) + 4.0

      #self.assertAlmostEqual(simulation_steady_state, correct_steady_state, delta=0.1)
