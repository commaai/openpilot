import math
import numpy as np
from collections import deque

from cereal import log
from openpilot.common.numpy_fast import interp
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.pid import PIDController
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]


class NanoFFModel:
  def __init__(self, temperature=1.0):
    self.w_1 = [[3.1900432109832764, -0.37573888897895813, -1.3121652603149414, 2.6689391136169434, 1.5531493425369263, 1.7823995351791382, 3.4191055297851562, 1.7675808668136597], [-0.7232083678245544, -0.06553429365158081, -0.05937732011079788, -0.6986712217330933, -0.18061856925487518, -0.4962347149848938, -0.8380135893821716, -0.3662603199481964], [-0.03277115896344185, 0.03201576694846153, -0.025812357664108276, 0.06850704550743103, 0.4274134039878845, -0.004680467303842306, -0.05513480678200722, 0.015983834862709045], [-0.30559229850769043, -0.27125728130340576, -0.39627915620803833, 0.06569751352071762, 0.26363715529441833, 0.06277585029602051, -0.38928478956222534, 0.10859646648168564], [1.2827545404434204, 1.7096779346466064, 0.7163336277008057, 0.8644662499427795, -0.39612334966659546, -0.03487994521856308, 1.605427622795105, -0.1349191665649414], [0.9179061055183411, 2.3207454681396484, 1.3605477809906006, 0.4095320403575897, -0.8882166743278503, -0.500495970249176, 1.1497553586959839, -0.5403992533683777], [0.35833898186683655, 2.8853394985198975, 1.9096674919128418, -0.06141189485788345, -1.646935224533081, -1.015617847442627, 0.5076309442520142, -1.0497525930404663], [-0.6210983991622925, -0.12343478947877884, -0.03978592902421951, -0.6785944104194641, -0.13170389831066132, -0.5560221672058105, -0.8075632452964783, -0.3575599193572998], [-0.5296007990837097, -0.389423668384552, -0.26156967878341675, -0.42312613129615784, 0.12104514241218567, -0.2939698100090027, -0.6980560421943665, -0.16298672556877136], [-0.2567983567714691, -0.5941768884658813, -0.5542563796043396, -0.2837754189968109, 0.4007265269756317, -0.1560964435338974, -0.36720964312553406, 0.04063839092850685], [0.012145349755883217, 0.18805164098739624, 0.0614439956843853, -0.07154802978038788, 0.5043861865997314, -0.049626994878053665, -0.05966878682374954, -0.10735032707452774], [0.01650974527001381, 0.001830110209994018, -0.04924454540014267, -0.05984148010611534, 0.5003781914710999, -0.04049041122198105, -0.06777270138263702, -0.016508804634213448], [-0.06617162376642227, 0.029375631362199783, -0.02804550901055336, 0.030467022210359573, 0.42632168531417847, -0.024852434173226357, -0.0030907196924090385, -0.16551297903060913], [-0.32580259442329407, -0.24488748610019684, -0.32683396339416504, -0.01919000968337059, 0.12490897625684738, 0.17895494401454926, -0.34487688541412354, 0.11210019886493683], [-0.2696765661239624, -0.3028377294540405, -0.24527736008167267, -0.08745969831943512, 0.2497100532054901, 0.07265375554561615, -0.37013161182403564, 0.17590118944644928], [-0.2743346691131592, -0.16069960594177246, -0.22537817060947418, -0.043523821979761124, 0.14299538731575012, 0.06332743912935257, -0.3213661313056946, 0.10957325249910355]]  # noqa: E501
    self.b_1 = [-0.8718048334121704, -0.5246109366416931, -0.30594491958618164, -0.590533435344696, -0.3367057740688324, -0.3095954954624176, -0.8740265369415283, -0.3223787844181061]  # noqa: E501
    self.w_2 = [[0.8718569874763489, 0.980969250202179, -1.0938196182250977, 0.5162032842636108], [-0.83515465259552, 0.02803085558116436, 1.0227752923965454, -0.3231082558631897], [-1.2006933689117432, -0.1881900131702423, 1.3541123867034912, -1.001417875289917], [0.5682749152183533, 0.27409738302230835, -0.8192962408065796, -0.03731514886021614], [0.5748910903930664, -0.007078315131366253, -0.3542194366455078, 0.5303947329521179], [0.6065377593040466, -0.4995797276496887, -0.8026412129402161, 0.3358980417251587], [1.2657983303070068, 1.0652618408203125, -1.3863341808319092, 0.6090320348739624], [0.5388105511665344, -0.06839455664157867, -0.7005999088287354, 0.5903877019882202]]  # noqa: E501
    self.b_2 = [-0.5377247333526611, -0.05827830359339714, 0.6316986083984375, -0.016132982447743416]
    self.w_3 = [[1.0667357444763184, 0.13138769567012787], [0.22754108905792236, -1.3573530912399292], [-1.8196125030517578, -2.071134567260742], [0.5297592878341675, -0.492388516664505]]  # noqa: E501
    self.b_3 = [0.12318920344114304, -0.6744083762168884]

    self.input_norm_mat = np.array([[-3.0, 3.0], [-3.0, 3.0], [0.0, 30.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [0.0, 30.0], [0.0, 30.0], [0.0, 30.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]])  # noqa: E501
    self.output_norm_mat = np.array([-1.0, 1.0])
    self.temperature = temperature

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def forward(self, x):
    # if x.ndim == 1:
    #   x = x.reshape(1, -1)
    assert x.ndim == 1
    x = (x - self.input_norm_mat[:, 0]) / (self.input_norm_mat[:, 1] - self.input_norm_mat[:, 0])
    x = self.sigmoid(np.dot(x, self.w_1) + self.b_1)
    x = self.sigmoid(np.dot(x, self.w_2) + self.b_2)
    x = np.dot(x, self.w_3) + self.b_3
    return x

  def predict(self, x):
    x = self.forward(np.array(x))
    pred = np.random.laplace(x[0], np.exp(x[1]) / self.temperature)
    pred = pred * (self.output_norm_mat[1] - self.output_norm_mat[0]) + self.output_norm_mat[0]
    return float(pred)


class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki,
                             k_f=self.torque_params.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.ff_model = NanoFFModel(temperature=10.0)
    self.history = {key: deque([0, 0, 0], maxlen=3) for key in ["lataccel", "roll_compensation", "vego", "aego"]}
    self.history_counter = 0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update(self, active, CS, VM, params, steer_limited, desired_curvature, llk):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    roll_compensation = math.sin(params.roll) * ACCELERATION_DUE_TO_GRAVITY

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      if self.use_steering_angle:
        actual_curvature = actual_curvature_vm
        # curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      else:
        actual_curvature_llk = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        actual_curvature = interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_llk])
        # curvature_deadzone = 0.0
      desired_lateral_accel = desired_curvature * CS.vEgo ** 2

      # desired rate is the desired rate of change in the setpoint, not the absolute desired curvature
      # desired_lateral_jerk = desired_curvature_rate * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2

      low_speed_factor = interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)**2
      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature

      # lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2
      # gravity_adjusted_lateral_accel = desired_lateral_accel - roll_compensation
      # torque_from_setpoint = self.torque_from_lateral_accel(setpoint, self.torque_params, setpoint,
      #                                                lateral_accel_deadzone, friction_compensation=False)
      # torque_from_measurement = self.torque_from_lateral_accel(measurement, self.torque_params, measurement,
      #                                                lateral_accel_deadzone, friction_compensation=False)
      # pid_log.error = torque_from_setpoint - torque_from_measurement
      # ff = self.torque_from_lateral_accel(gravity_adjusted_lateral_accel, self.torque_params,
      #                                     desired_lateral_accel - actual_lateral_accel,
      #                                     lateral_accel_deadzone, friction_compensation=True)

      state_vector = [roll_compensation, CS.vEgo, CS.aEgo]
      history_state_vector = list(self.history["lataccel"]) + list(self.history["roll_compensation"]) + list(self.history["vego"]) + list(self.history["aego"])
      torque_from_setpoint = self.ff_model.predict([setpoint] + state_vector + history_state_vector)
      torque_from_measurement = self.ff_model.predict([measurement] + state_vector + history_state_vector)
      pid_log.error = torque_from_setpoint - torque_from_measurement
      ff = self.ff_model.predict([desired_lateral_accel] + state_vector + history_state_vector)

      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      output_torque = self.pid.update(pid_log.error,
                                      feedforward=ff,
                                      speed=CS.vEgo,
                                      freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.d = self.pid.d
      pid_log.f = self.pid.f
      pid_log.output = -output_torque
      pid_log.actualLateralAccel = actual_lateral_accel
      pid_log.desiredLateralAccel = desired_lateral_accel
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited)

    if self.history_counter % 10 == 0:
      self.history["lataccel"].append(actual_curvature_vm * CS.vEgo ** 2)
      self.history["roll_compensation"].append(roll_compensation)
      self.history["vego"].append(CS.vEgo)
      self.history["aego"].append(CS.aEgo)

    self.history_counter = (self.history_counter + 1) % 10

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
