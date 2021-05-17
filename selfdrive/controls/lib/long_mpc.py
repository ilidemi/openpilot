from datetime import datetime
from enum import Enum
import os
import math
from threading import Thread
from queue import Full, Queue
import struct
import subprocess
import time

import cereal.messaging as messaging
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from selfdrive.controls.lib.dynamic_follow import DynamicFollow

LOG_MPC = os.environ.get('LOG_MPC', False)


class InputEvent(Enum):
  SHORT_PRESS = 1
  LONG_PRESS = 2


def input_loop(input_queue):
  log_f = open('/data/openpilot-patch/input_log.txt', 'a')

  try:
    button_file = open('/dev/input/event0', 'rb')
    is_pressed = None
    state_change_time = time.monotonic()

    short_press_range = (0.7, 2.)
    long_press_range = (3., 5.)

    while True:
      button_data = button_file.read(24)
      button_fields = struct.unpack('4IHHI', button_data)
      if button_fields[5] != 114: # Volume down
        continue
      new_state_change_time = time.monotonic()
      time_in_state = new_state_change_time - state_change_time
      new_is_pressed = button_fields[6] == 1

      if is_pressed is None:
        pass
      elif not is_pressed and new_is_pressed:
        pass
      elif is_pressed and not new_is_pressed:
        # TODO: remove inline dims
        if short_press_range[0] <= time_in_state <= short_press_range[1]:
          input_queue.put(InputEvent.SHORT_PRESS)
          dim(1.0)
        elif long_press_range[0] <= time_in_state <= long_press_range[1]:
          input_queue.put(InputEvent.LONG_PRESS)
          dim(3.0)
      else: # Two pressed or two depressed events in a row
        log_f.write(f"{datetime.now()} is_pressed:{is_pressed} new_is_pressed:{new_is_pressed} duration:{time_in_state}\n")
        log_f.flush()
      is_pressed = new_is_pressed
      state_change_time = new_state_change_time
  except Exception as e:
    log_f.write(f"{datetime.now()} Input loop exception: {e}\n")
    log_f.flush()
    subprocess.Popen(['python', '/data/openpilot-patch/util/error.py'])


class OutputEvent(Enum):
  SHORT_DIM = 1
  LONG_DIM = 2

def dim(duration):
  brightness_path = '/sys/class/leds/lcd-backlight/brightness'
  with open(brightness_path, 'w') as brightness_f:
    brightness_f.write('63')
  time.sleep(duration)
  with open(brightness_path, 'w') as brightness_f:
    brightness_f.write('127')


def output_loop(output_queue):
  log_f = open('/data/openpilot-patch/output_log.txt', 'a')

  try:
    while True:
      output_event = output_queue.get(block=True)
      if output_event == OutputEvent.SHORT_DIM:
        dim(1.0)
      elif output_event == OutputEvent.LONG_DIM:
        dim(3.0)
  except Exception as e:
    log_f.write(f"{datetime.now()} Output loop exception: {e}\n")
    log_f.flush()
    subprocess.Popen(['python', '/data/openpilot-patch/util/error.py'])


class LongitudinalMpc():
  def __init__(self, mpc_id, TR_override=None):
    self.mpc_id = mpc_id

    self.dynamic_follow = DynamicFollow(mpc_id)
    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False

    self.last_cloudlog_t = 0.0
    self.n_its = 0
    self.duration = 0

    self.TR_override = TR_override

    self.input_queue = Queue()
    self.input_thread = Thread(target=input_loop, args=(self.input_queue,))
    self.input_thread.start()

    self.output_queue = Queue()
    self.output_thread = Thread(target=output_loop, args=(self.output_queue,))
    self.output_thread.start()

  def publish(self, pm):
    if LOG_MPC:
      qp_iterations = max(0, self.n_its)
      dat = messaging.new_message('liveLongitudinalMpc')
      dat.liveLongitudinalMpc.xEgo = list(self.mpc_solution[0].x_ego)
      dat.liveLongitudinalMpc.vEgo = list(self.mpc_solution[0].v_ego)
      dat.liveLongitudinalMpc.aEgo = list(self.mpc_solution[0].a_ego)
      dat.liveLongitudinalMpc.xLead = list(self.mpc_solution[0].x_l)
      dat.liveLongitudinalMpc.vLead = list(self.mpc_solution[0].v_l)
      dat.liveLongitudinalMpc.cost = self.mpc_solution[0].cost
      dat.liveLongitudinalMpc.aLeadTau = self.a_lead_tau
      dat.liveLongitudinalMpc.qpIterations = qp_iterations
      dat.liveLongitudinalMpc.mpcId = self.mpc_id
      dat.liveLongitudinalMpc.calculationTime = self.duration
      pm.send('liveLongitudinalMpc', dat)

  def setup_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.mpc_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, CS, lead):
    try:
      input_event = self.input_queue.get_nowait()
    
      if input_event == InputEvent.LONG_PRESS:
        self.TR_override = 1.8 if self.TR_override is None else None
      
      # Output current status
      output_event = OutputEvent.LONG_DIM if self.TR_override is None else OutputEvent.SHORT_DIM
      try:
        self.output_queue.put_nowait(output_event)
      except Full:
        log_f = open('/data/openpilot-patch/mpc_update_log.txt', 'a')
        log_f.write(f"{datetime.now()} Output queue is full\n")
        log_f.flush()
        subprocess.Popen(['python', '/data/openpilot-patch/util/error.py'])
    except TimeoutError:
      pass

    v_ego = CS.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = lead.aLeadTau
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.libmpc.init_with_simulation(self.v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.dynamic_follow.update_lead(v_lead, a_lead, x_lead, lead.status, self.new_lead)
      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.dynamic_follow.update_lead(new_lead=self.new_lead)
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU

    if self.TR_override:
      TR = self.TR_override
    else:
      TR = self.dynamic_follow.update(CS, self.libmpc)  # update dynamic follow

    # Calculate mpc
    t = sec_since_boot()
    self.n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead, TR)
    self.duration = int((sec_since_boot() - t) * 1e9)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]

    # Reset if NaN or goes through lead car
    crashing = any(lead - ego < -50 for (lead, ego) in zip(self.mpc_solution[0].x_l, self.mpc_solution[0].x_ego))
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)
    backwards = min(self.mpc_solution[0].v_ego) < -0.01

    if ((backwards or crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.mpc_id, backwards, crashing, nans))

      self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                       MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0
      self.v_mpc = v_ego
      self.a_mpc = CS.aEgo
      self.prev_lead_status = False
