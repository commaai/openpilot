from common.numpy_fast import clip


def create_lkas_command(packer, angle_deg: float, curvature: float):
  """
  Creates a CAN message for the Ford LKAS Command.

  This command can apply "Lane Keeping Aid" manoeuvres, which are subject to the
  PSCM lockout.

  Frequency is 20Hz.
  """

  values = {
    "LkaDrvOvrrd_D_Rq": 0,              # driver override level? [0|3]
    "LkaActvStats_D2_Req": 0,           # action [0|7]
    "LaRefAng_No_Req": angle_deg,       # angle [-102.4|102.3] degrees
    "LaRampType_B_Req": 0,              # Ramp speed: 0=Smooth, 1=Quick
    "LaCurvature_No_Calc": curvature,   # curvature [-0.01024|0.01023] 1/meter
    "LdwActvStats_D_Req": 0,            # LDW status [0|7]
    "LdwActvIntns_D_Req": 0,            # LDW intensity [0|3], shake alert strength
  }
  return packer.make_can_msg("Lane_Assist_Data1", 0, values)


def create_tja_command(packer, lca_rq: int, ramp_type: int, precision: int, path_offset: float, path_angle: float, curvature_rate: float, curvature: float):
  """
  Creates a CAN message for the Ford TJA/LCA Command.

  This command can apply "Lane Centering" manoeuvres: continuous lane centering
  for traffic jam assist and highway driving. It is not subject to the PSCM
  lockout.

  The PSCM should be configured to accept TJA/LCA commands before these
  commands will be processed. This can be done using tools such as Forscan.

  Frequency is 20Hz.
  """

  values = {
    "LatCtlRng_L_Max": 0,                                                   # Unknown [0|126] meter
    "HandsOffCnfm_B_Rq": 0,                                                 # Unknown: 0=Inactive, 1=Active [0|1]
    "LatCtl_D_Rq": lca_rq,                                                  # Mode: 0=None, 1=ContinuousPathFollowing, 2=InterventionLeft, 3=InterventionRight, 4-7=NotUsed [0|7]
    "LatCtlRampType_D_Rq": ramp_type,                                       # Ramp speed: 0=Slow, 1=Medium, 2=Fast, 3=Immediate [0|3]
    "LatCtlPrecision_D_Rq": precision,                                      # Precision: 0=Comfortable, 1=Precise, 2/3=NotUsed [0|3]
    "LatCtlPathOffst_L_Actl": clip(path_offset, -5.12, 5.11),               # Path offset [-5.12|5.11] meter
    "LatCtlPath_An_Actl": clip(path_angle, -0.5, 0.5235),                   # Path angle [-0.5|0.5235] radians
    "LatCtlCurv_NoRate_Actl": clip(curvature_rate, -0.001024, 0.00102375),  # Curvature rate [-0.001024|0.00102375] 1/meter^2
    "LatCtlCurv_No_Actl": clip(curvature, -0.02, 0.02094),                  # Curvature [-0.02|0.02094] 1/meter
  }
  return packer.make_can_msg("LateralMotionControl", 0, values)


def create_lkas_ui_command(packer, main_on: bool, enabled: bool, steer_alert: bool, stock_values: dict):
  """
  Creates a CAN message for the Ford IPC IPMA/LKAS status.

  Show the LKAS status with the "driver assist" lines in the IPC.

  Stock functionality is maintained by passing through unmodified signals.

  Frequency is 1Hz.
  """

  # LaActvStats_D_Dsply
  # TODO: get LDW state from OP
  if enabled:
    lines = 6
  elif main_on:
    lines = 0
  else:
    lines = 30

  values = {
    **stock_values,
    "LaActvStats_D_Dsply": lines,                           # LKAS status (lines) [0|31]
    "LaHandsOff_D_Dsply": 2 if steer_alert else 0,          # 0=HandsOn, 1=Level1 (w/o chime), 2=Level2 (w/ chime), 3=Suppressed
  }
  return packer.make_can_msg("IPMA_Data", 0, values)


def create_acc_ui_command(packer, main_on: bool, enabled: bool, stock_values: dict):
  """
  Creates a CAN message for the Ford IPC adaptive cruise, forward collision
  warning and traffic jam assist status.

  Stock functionality is maintained by passing through unmodified signals.

  Frequency is 20Hz.
  """

  values = {
    **stock_values,
    "Tja_D_Stat": 2 if enabled else (1 if main_on else 0),  # TJA status: 0=Off, 1=Standby, 2=Active, 3=InterventionLeft, 4=InterventionRight, 5=WarningLeft, 6=WarningRight, 7=NotUsed [0|7]
  }
  return packer.make_can_msg("ACCDATA_3", 0, values)


def spam_cancel_button(packer, cancel=1):
  """
  Creates a CAN message for the Ford SCCM buttons/switches.

  Includes cruise control buttons, turn lights and more.
  """

  values = {
    "CcAslButtnCnclPress": cancel,  # CC cancel button
  }
  return packer.make_can_msg("Steering_Data_FD1", 0, values)
