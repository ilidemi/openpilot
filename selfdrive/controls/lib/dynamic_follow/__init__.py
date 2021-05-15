import cereal.messaging as messaging
from common.realtime import sec_since_boot
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from common.numpy_fast import interp, clip
from selfdrive.config import Conversions as CV

from selfdrive.controls.lib.dynamic_follow.support import LeadData, CarData, dfData
travis = False


class DistanceModController:
  def __init__(self, k_i, k_d, x_clip, mods):
    self._rate = 1 / 20.

    self._k_i = k_i
    self._k_d = k_d
    self._to_clip = x_clip  # reaches this with v_rel=3.5 mph for 4 seconds
    self._mods = mods

    self.i = 0  # never resets, even when new lead
    self.last_error = 0

  def update(self, error):
    """
    Relative velocity is a good starting point
    Returns: Multiplier for final y_dist output
    """

    d = self._k_d * (error - self.last_error)
    if d < 0:  # only add if it will add distance
      self.i += d

    self.i += error * self._rate * self._k_i
    self.i = clip(self.i, self._to_clip[0], self._to_clip[-1])  # clip to reasonable range
    self._slow_reset()  # slowly reset from max to 0

    fact = interp(self.i, self._to_clip, self._mods)
    self.last_error = float(error)

    return fact

  def _slow_reset(self):
    if abs(self.i) > 0.01:  # oscillation starts around 0.006
      reset_time = 15  # in x seconds i goes from max to 0
      sign = 1 if self.i > 0 else -1
      self.i -= sign * max(self._to_clip) / (reset_time / self._rate)


class DynamicFollow:
  def __init__(self, mpc_id):
    self.mpc_id = mpc_id
    self.dmc_v_rel = DistanceModController(k_i=0.042, k_d=0.08, x_clip=[-1, 0, 0.66], mods=[1.15, 1., 0.95])
    self.dmc_a_rel = DistanceModController(k_i=0.042 * 1.05, k_d=0.08, x_clip=[-1, 0, 0.33], mods=[1.15, 1., 0.98])  # a_lead loop is 5% faster

    # Dynamic follow variables
    self.default_TR = 1.8
    self.TR = 1.8
    self.v_ego_retention = 2.5
    self.v_rel_retention = 1.75

    self.sng_TR = 1.8  # reacceleration stop and go TR
    self.sng_speed = 18.0 * CV.MPH_TO_MS

    self._setup_changing_variables()

  def _setup_changing_variables(self):
    self.TR = self.default_TR

    self.sng = False
    self.car_data = CarData()
    self.lead_data = LeadData()
    self.df_data = dfData()  # dynamic follow data

    self.last_cost = 0.0

  def update(self, CS, libmpc):
    self._update_car(CS)

    if not self.lead_data.status:
      self.TR = self.default_TR
    else:
      self._store_df_data()
      self.TR = self._get_TR()

    if not travis:
      self._change_cost(libmpc)

    return self.TR

  def _change_cost(self, libmpc):
    TRs = [0.9, 1.8, 2.7]
    costs = [1., 0.1, 0.01]
    cost = interp(self.TR, TRs, costs)

    if self.last_cost != cost:
      libmpc.change_costs(MPC_COST_LONG.TTC, cost, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)  # todo: jerk is the derivative of acceleration, could tune that
      self.last_cost = cost

  def _store_df_data(self):
    cur_time = sec_since_boot()
    # Store custom relative accel over time
    if self.lead_data.status:
      if self.lead_data.new_lead:
        self.df_data.v_rels = []  # reset when new lead
      else:
        self.df_data.v_rels = self._remove_old_entries(self.df_data.v_rels, cur_time, self.v_rel_retention)
      self.df_data.v_rels.append({'v_ego': self.car_data.v_ego, 'v_lead': self.lead_data.v_lead, 'time': cur_time})

    # Store our velocity for better sng
    self.df_data.v_egos = self._remove_old_entries(self.df_data.v_egos, cur_time, self.v_ego_retention)
    self.df_data.v_egos.append({'v_ego': self.car_data.v_ego, 'time': cur_time})

  @staticmethod
  def _remove_old_entries(lst, cur_time, retention):
    return [sample for sample in lst if cur_time - sample['time'] <= retention]

  def _get_TR(self):
    x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
    y_dist = [1.3781, 1.3791, 1.3457, 1.3134, 1.3145, 1.318, 1.3485, 1.257, 1.2, 1.2, 1.2, 1.2]

    v_rel_dist_factor = self.dmc_v_rel.update(self.lead_data.v_lead - self.car_data.v_ego)
    a_lead_dist_factor = self.dmc_a_rel.update(self.lead_data.a_lead - self.car_data.a_ego)

    TR = interp(self.car_data.v_ego, x_vel, y_dist)
    TR *= v_rel_dist_factor
    TR *= a_lead_dist_factor

    if self.car_data.v_ego > self.sng_speed:  # keep sng distance until we're above sng speed again
      self.sng = False

    if (self.car_data.v_ego >= self.sng_speed or self.df_data.v_egos[0]['v_ego'] >= self.car_data.v_ego) and not self.sng:
      # if above 15 mph OR we're decelerating to a stop, keep shorter TR. when we reaccelerate, use sng_TR and slowly decrease
      TR = interp(self.car_data.v_ego, x_vel, y_dist)
    else:  # this allows us to get closer to the lead car when stopping, while being able to have smooth stop and go when reaccelerating
      self.sng = True
      x = [self.sng_speed * 0.7, self.sng_speed]  # decrease TR between 12.6 and 18 mph from 1.8s to defined TR above at 18mph while accelerating
      y = [self.sng_TR, interp(self.sng_speed, x_vel, y_dist)]
      TR = interp(self.car_data.v_ego, x, y)

    return float(clip(TR, 1.0, 2.7))

  def update_lead(self, v_lead=None, a_lead=None, x_lead=None, status=False, new_lead=False):
    self.lead_data.v_lead = v_lead
    self.lead_data.a_lead = a_lead
    self.lead_data.x_lead = x_lead

    self.lead_data.status = status
    self.lead_data.new_lead = new_lead

  def _update_car(self, CS):
    self.car_data.v_ego = CS.vEgo
    self.car_data.a_ego = CS.aEgo

    self.car_data.left_blinker = CS.leftBlinker
    self.car_data.right_blinker = CS.rightBlinker
    self.car_data.cruise_enabled = CS.cruiseState.enabled
