"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from opendbc.car import Bus, structs
from opendbc.car.can_definitions import CanRecvCallable, CanSendCallable
from opendbc.car.car_helpers import can_fingerprint
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.hyundai.radar_interface import RADAR_START_ADDR
from opendbc.car.hyundai.values import HyundaiFlags, DBC as HYUNDAI_DBC
from opendbc.sunnypilot.car.hyundai.longitudinal.helpers import LongitudinalTuningType
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import get_nn_model_path

import openpilot.system.sentry as sentry


def log_fingerprint(CP: structs.CarParams) -> None:
  if CP.carFingerprint == "MOCK":
    sentry.capture_fingerprint_mock()
  else:
    sentry.capture_fingerprint(CP.carFingerprint, CP.brand)

def _initialize_custom_longitudinal_tuning(CI: CarInterfaceBase, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                                           params: Params = None) -> None:
  if params is None:
    params = Params()

  # Hyundai Custom Longitudinal Tuning
  if CP.brand == 'hyundai':
    hyundai_longitudinal_tuning = int(params.get("HyundaiLongitudinalTuning", encoding="utf8") or 0)
    if hyundai_longitudinal_tuning == LongitudinalTuningType.DYNAMIC:
      CP_SP.flags |= HyundaiFlagsSP.LONG_TUNING_DYNAMIC.value
    if hyundai_longitudinal_tuning == LongitudinalTuningType.PREDICTIVE:
      CP_SP.flags |= HyundaiFlagsSP.LONG_TUNING_PREDICTIVE.value

  CP_SP = CI.get_longitudinal_tuning_sp(CP, CP_SP)


def _initialize_neural_network_lateral_control(CI: CarInterfaceBase, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                                               params: Params = None, enabled: bool = False) -> None:
  if params is None:
    params = Params()

  nnlc_model_path, nnlc_model_name, exact_match = get_nn_model_path(CP)

  if nnlc_model_name == "MOCK":
    cloudlog.error({"nnlc event": "car doesn't match any Neural Network model"})

  if nnlc_model_name != "MOCK" and CP.steerControlType != structs.CarParams.SteerControlType.angle:
    enabled = params.get_bool("NeuralNetworkLateralControl")

  if enabled:
    CI.configure_torque_tune(CP.carFingerprint, CP.lateralTuning)

  CP_SP.neuralNetworkLateralControl.model.path = nnlc_model_path
  CP_SP.neuralNetworkLateralControl.model.name = nnlc_model_name
  CP_SP.neuralNetworkLateralControl.fuzzyFingerprint = not exact_match


def _initialize_radar_tracks(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: Params = None) -> None:
  if params is None:
    params = Params()

  if CP.brand == 'hyundai':
    if CP.flags & HyundaiFlags.MANDO_RADAR and CP.radarUnavailable:
      # Having this automatic without a toggle causes a weird process replay diff because
      # somehow it sees fewer logs than intended
      if params.get_bool("HyundaiRadarTracksToggle"):
        CP_SP.flags |= HyundaiFlagsSP.ENABLE_RADAR_TRACKS.value
        if params.get_bool("HyundaiRadarTracks"):
          CP.radarUnavailable = False


def setup_interfaces(CI: CarInterfaceBase, params: Params = None) -> None:
  CP = CI.CP
  CP_SP = CI.CP_SP

  _initialize_custom_longitudinal_tuning(CI, CP, CP_SP, params)
  _initialize_neural_network_lateral_control(CI, CP, CP_SP, params)
  _initialize_radar_tracks(CP, CP_SP, params)


def _enable_radar_tracks(CP: structs.CarParams, CP_SP: structs.CarParamsSP, can_recv: CanRecvCallable,
                        params: Params) -> None:
  if CP.brand == 'hyundai':
    if CP_SP.flags & HyundaiFlagsSP.ENABLE_RADAR_TRACKS:
      can_recv()
      _, fingerprint = can_fingerprint(can_recv)
      radar_unavailable = RADAR_START_ADDR not in fingerprint[1] or Bus.radar not in HYUNDAI_DBC[CP.carFingerprint]

      radar_tracks = params.get_bool("HyundaiRadarTracks")
      radar_tracks_persistent = params.get_bool("HyundaiRadarTracksPersistent")

      params.put_bool_nonblocking("HyundaiRadarTracksConfirmed", radar_tracks)

      if not radar_tracks_persistent:
        params.put_bool_nonblocking("HyundaiRadarTracks", not radar_unavailable)
        params.put_bool_nonblocking("HyundaiRadarTracksPersistent", True)


def init_interfaces(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: Params,
                                can_recv: CanRecvCallable, can_send: CanSendCallable):
  _enable_radar_tracks(CP, CP_SP, can_recv, params)
