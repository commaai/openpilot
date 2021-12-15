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


class dfProfiles:
  traffic = 0
  stock = 1
  relaxed = 2
  auto = 3
  to_profile = {0: 'traffic', 1: 'stock', 2: 'relaxed', 3: 'auto'}
  to_idx = {v: k for k, v in to_profile.items()}

  default = relaxed
