class LeadData:
  v_lead = None
  x_lead = None
  a_lead = None
  status = False
  new_lead = False


class CarData:
  v_ego = 0.0
  a_ego = 0.0

  left_blinker = False
  right_blinker = False
  cruise_enabled = True


class dfData:
  v_egos = []
  v_rels = []
