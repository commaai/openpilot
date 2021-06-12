from selfdrive.car.toyota.values import CAR as CAR_TOYOTA, MIN_ACC_SPEED
from selfdrive.car.honda.values import CAR as CAR_HONDA
from common.numpy_fast import clip, interp


class DynamicGas:
  def __init__(self, CP, candidate):
    self.toyota_candidates = [attr for attr in dir(CAR_TOYOTA) if not attr.startswith("__")]
    self.honda_candidates = [attr for attr in dir(CAR_HONDA) if not attr.startswith("__")]

    self.candidate = candidate
    self.CP = CP
    self.set_profile()

    self.lead_data = {'v_rel': None, 'a_lead': None, 'x_lead': None, 'status': False}
    self.mpc_TR = 1.8
    self.blinker_status = False
    self.gas_pressed = False

  def update(self, CS, extra_params):
    v_ego = CS.vEgo
    self.handle_passable(CS, extra_params)

    # disable dynamic gas if car not supported OR if pedal and using stock ACC system with acceleration instead of gas
    if not self.supported_car or (self.CP.enableGasInterceptor and v_ego > MIN_ACC_SPEED):
      return float(interp(v_ego, self.CP.gasMaxBP, self.CP.gasMaxV))

    gas = interp(v_ego, self.gasMaxBP, self.gasMaxV)
    if self.lead_data['status']:  # if lead
      if v_ego <= 8.9408:  # if under 20 mph
        x = [0.0, 0.24588812499999999, 0.432818589, 0.593044697, 0.730381365, 1.050833588, 1.3965, 1.714627481]  # relative velocity mod
        y = [0.9901, 0.905, 0.8045, 0.625, 0.431, 0.2083, .0667, 0]
        gas_mod = -(gas * interp(self.lead_data['v_rel'], x, y))

        x = [0.44704, 1.1176, 1.34112]  # lead accel mod
        y = [1.0, 0.75, 0.625]  # maximum we can reduce gas_mod is 40 percent (never increases mod)
        gas_mod *= interp(self.lead_data['a_lead'], x, y)

        x = [6.1, 9.15, 15.24]  # as lead gets further from car, lessen gas mod/reduction
        y = [1.0, 0.8, 0.0]
        gas_mod *= interp(self.lead_data['x_lead'], x, y)

        if not self.CP.enableGasInterceptor:  # this will hopefuly let TSS2 use dynamic gas, need to tune
          gas_mod *= 0.33
        new_gas = gas + gas_mod

        x = [1.78816, 6.0, 8.9408]  # slowly ramp mods down as we approach 20 mph
        y = [new_gas, (new_gas * 0.6 + gas * 0.4), gas]
        gas = interp(v_ego, x, y)
      else:
        x = [-3.12928, -1.78816, -0.89408, 0, 0.33528, 1.78816, 2.68224]  # relative velocity mod
        y = [-gas * 0.2625, -gas * 0.18, -gas * 0.12, 0.0, gas * 0.075, gas * 0.135, gas * 0.19]
        gas_mod = interp(self.lead_data['v_rel'], x, y)

        current_TR = self.lead_data['x_lead'] / v_ego
        x = [self.mpc_TR - 0.22, self.mpc_TR, self.mpc_TR + 0.2, self.mpc_TR + 0.4]
        y = [-gas_mod * 0.15, 0.0, gas_mod * 0.25, gas_mod * 0.4]
        gas_mod -= interp(current_TR, x, y)

        gas += gas_mod

        if self.blinker_status:
          x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
          y = [1.0, 1.2875, 1.4625]
          gas *= interp(v_ego, x, y)

    return float(clip(gas, 0.0, 1.0))

  def set_profile(self):
    self.supported_car = True  # all pedal cars and some tuned non-pedal cars are supported
    if self.CP.enableGasInterceptor:
      x = [0.0, 1.4082, 2.8031, 4.2266, 5.3827, 6.1656, 7.2478, 8.2831, 10.2447, 12.964, 15.423, 18.119, 20.117, 24.4661, 29.0581, 32.7101, 35.7633]
      y = [0.219, 0.225, 0.237, 0.255, 0.279, 0.3, 0.339, 0.361, 0.376, 0.385, 0.394, 0.408, 0.421, 0.459, 0.513, 0.569, 0.629]  # This is the TSS1 Corolla tune, use by default for all pedal cars
      if self.candidate == CAR_TOYOTA.PRIUS:
        y = [0.3, 0.304, 0.315, 0.342, 0.365, 0.386, 0.429, 0.454, 0.472, 0.48, 0.489, 0.421, 0.432, 0.480, 0.55, 0.621, 0.7]
      elif self.candidate == CAR_TOYOTA.RAV4:
        y = [0.257, 0.261, 0.271, 0.286, 0.307, 0.327, 0.365, 0.389, 0.405, 0.415, 0.428, 0.439, 0.452, 0.495, 0.554, 0.614, 0.679]
    else:
      x = [0.0, 1.4082, 2.80311, 4.22661, 5.38271, 6.16561, 7.24781, 8.28308, 10.24465, 12.96402, 15.42303, 18.11903, 20.11703, 24.46614, 29.05805, 32.71015, 35.76326]
      y = [0.234, 0.237, 0.246, 0.26, 0.279, 0.297, 0.332, 0.354, 0.368, 0.377, 0.389, 0.399, 0.411, 0.45, 0.504, 0.558, 0.617]
      if self.candidate in [CAR_TOYOTA.PRIUS_TSS2, CAR_TOYOTA.PRIUS]:
        y = [0.35587, 0.46747, 0.41816, 0.33261, 0.27844, 0.2718, 0.28184, 0.29106, 0.29785, 0.297, 0.29658, 0.30308, 0.31354, 0.34922, 0.39767, 0.44527, 0.4984]
      elif self.candidate in [CAR_TOYOTA.RAV4_TSS2, CAR_TOYOTA.RAV4H_TSS2]:
        y = [0.35587, 0.46747, 0.41816, 0.33261, 0.27844, 0.2718, 0.28396, 0.29537, 0.30647, 0.31161, 0.3168, 0.3272, 0.34, 0.3824, 0.44, 0.4968, 0.56]
      else:
        self.supported_car = False  # make this true to use default pedalless gasMax tune for your car without a profile

    self.gasMaxBP, self.gasMaxV = x, y

  def handle_passable(self, CS, passable):
    self.blinker_status = CS.leftBlinker or CS.rightBlinker
    self.gas_pressed = CS.gasPressed
    self.lead_data['v_rel'] = passable['lead_one'].vRel
    self.lead_data['a_lead'] = passable['lead_one'].aLeadK
    self.lead_data['x_lead'] = passable['lead_one'].dRel
    self.lead_data['status'] = passable['has_lead']  # this fixes radarstate always reporting a lead, thanks to arne
    self.mpc_TR = passable['mpc_TR']
