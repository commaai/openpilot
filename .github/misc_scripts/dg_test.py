# from selfdrive.car.toyota.values import CAR as CAR_TOYOTA
# from selfdrive.car.honda.values import CAR as CAR_HONDA
from common.numpy_fast import clip, interp
from selfdrive.config import Conversions as CV
CAR_TOYOTA = ['Corolla']
CAR_HONDA = []

class CarParams:
  enableGasInterceptor = True
  gasMaxBP = [0., 9., 35]
  gasMaxV = [0.2, 0.5, 0.7]

class DynamicGas:
  def __init__(self, CP, candidate):
    self.toyota_candidates = [attr for attr in dir(CAR_TOYOTA) if not attr.startswith("__")]
    self.honda_candidates = [attr for attr in dir(CAR_HONDA) if not attr.startswith("__")]

    self.candidate = candidate
    self.CP = CP

  def update(self, v_ego, lead_data, mpc_TR, blinker_status):
    x, y = self.CP.gasMaxBP, self.CP.gasMaxV
    if self.CP.enableGasInterceptor:
      if self.candidate == "Corolla":
        x = [0.0, 1.4082, 2.8031, 4.2266, 5.3827, 6.1656, 7.2478, 8.2831, 10.2447, 12.964, 15.423, 18.119, 20.117, 24.4661, 29.0581, 32.7101, 35.7633]
        y = [0.215, 0.219, 0.23, 0.247, 0.271, 0.292, 0.335, 0.361, 0.38, 0.388, 0.396, 0.41, 0.421, 0.46, 0.513, 0.566, 0.624]

    if not x:  # if unsupported car, don't use dynamic gas
      # x, y = CP.gasMaxBP, CP.gasMaxV  # if unsupported car, use stock.
      return interp(v_ego, self.CP.gasMaxBP, self.CP.gasMaxV)
    gas = interp(v_ego, x, y)
    print('Original gas: {}'.format(round(gas, 4)))

    if lead_data['status']:  # if lead
      if v_ego <= 8.9408:  # if under 20 mph
        x = [0.0, 0.24588812499999999, 0.432818589, 0.593044697, 0.730381365, 1.050833588, 1.3965, 1.714627481]  # relative velocity mod
        y = [gas * 0.9901, gas * 0.905, gas * 0.8045, gas * 0.625, gas * 0.431, gas * 0.2083, gas * .0667, 0]
        gas_mod = -interp(lead_data['v_rel'], x, y)

        x = [0.44704, 1.1176, 1.34112]  # lead accel mod
        y = [1.0, 0.85, 0.75]  # maximum we can reduce gas_mod is 40 percent (never increases mod)
        gas_mod *= interp(lead_data['a_lead'], x, y)

        x = [0.6096, 3.048, 8, 10, 12]  # as lead gets further from car, lessen gas mod
        y = [1.0, 0.9, 0.65, 0.3, 0]
        gas_mod *= interp(lead_data['x_lead'], x, y)

        print('Gas mod: {}'.format(round(gas_mod, 4)))
        new_gas = gas + gas_mod
        print('New gas: {}'.format(round(new_gas, 4)))

        x = [1.78816, 6.0, 8.9408]  # slowly ramp mods down as we approach 20 mph
        y = [new_gas, (new_gas * 0.8 + gas * 0.2), gas]
        gas = interp(v_ego, x, y)
      else:
        x = [-3.12928, -1.78816, -0.89408, 0, 0.33528, 1.78816, 2.68224]  # relative velocity mod
        y = [-gas * 0.2625, -gas * 0.18, -gas * 0.12, 0.0, gas * 0.075, gas * 0.135, gas * 0.19]
        gas_mod = interp(lead_data['v_rel'], x, y)

        current_TR = lead_data['x_lead'] / v_ego
        x = [mpc_TR - 0.22, mpc_TR, mpc_TR + 0.2, mpc_TR + 0.4]
        y = [-gas_mod * 0.15, 0.0, gas_mod * 0.25, gas_mod * 0.4]
        gas_mod -= interp(current_TR, x, y)

        gas += gas_mod

        if blinker_status:
          x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
          y = [1.0, 1.275, 1.45]
          gas *= interp(v_ego, x, y)

    return clip(gas, 0.0, 1.0)


CP = CarParams()
dg = DynamicGas(CP, 'Corolla')

lead_data = {'status': True, 'a_lead': 1.1, 'v_rel': -10, 'x_lead': 20}
v_ego = 2.5 * CV.MPH_TO_MS
mpc_TR = 1.9
o = dg.update(v_ego, lead_data, mpc_TR, False)
