from cereal import car
from selfdrive.car.ford.values import CANBUS

HUDControl = car.CarControl.HUDControl


def create_lka_command(packer, angle_deg: float, curvature: float):
  """
  Creates a CAN message for the Ford LKAS Command.

  This command can apply "Lane Keeping Aid" manoeuvres, which are subject to the PSCM lockout.

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
  return packer.make_can_msg("Lane_Assist_Data1", CANBUS.main, values)


def create_tja_command(packer, lca_rq: int, ramp_type: int, precision: int, path_offset: float, path_angle: float, curvature_rate: float, curvature: float):
  """
  Creates a CAN message for the Ford TJA/LCA Command.

  This command can apply "Lane Centering" manoeuvres: continuous lane centering for traffic jam
  assist and highway driving. It is not subject to the PSCM lockout.

  Ford lane centering command uses a third order polynomial to describe the road centerline. The
  polynomial is defined by the following coefficients:
    c0: lateral offset between the vehicle and the centerline
    c1: heading angle between the vehicle and the centerline
    c2: curvature of the centerline
    c3: rate of change of curvature of the centerline
  As the PSCM combines this information with other sensor data, such as the vehicle's yaw rate and
  speed, the steering angle cannot be easily controlled.

  The PSCM should be configured to accept TJA/LCA commands before these commands will be processed.
  This can be done using tools such as Forscan.

  Frequency is 20Hz.
  """

  values = {
    "LatCtlRng_L_Max": 0,                                   # Unknown [0|126] meter
    "HandsOffCnfm_B_Rq": 0,                                 # Unknown: 0=Inactive, 1=Active [0|1]
    "LatCtl_D_Rq": lca_rq,                                  # Mode: 0=None, 1=ContinuousPathFollowing, 2=InterventionLeft, 3=InterventionRight, 4-7=NotUsed [0|7]
    "LatCtlRampType_D_Rq": ramp_type,                       # Ramp speed: 0=Slow, 1=Medium, 2=Fast, 3=Immediate [0|3]
    "LatCtlPrecision_D_Rq": precision,                      # Precision: 0=Comfortable, 1=Precise, 2/3=NotUsed [0|3]
    "LatCtlPathOffst_L_Actl": path_offset,                  # Path offset [-5.12|5.11] meter
    "LatCtlPath_An_Actl": path_angle,                       # Path angle [-0.5|0.5235] radians
    "LatCtlCurv_NoRate_Actl": curvature_rate,               # Curvature rate [-0.001024|0.00102375] 1/meter^2
    "LatCtlCurv_No_Actl": curvature,                        # Curvature [-0.02|0.02094] 1/meter
  }
  return packer.make_can_msg("LateralMotionControl", CANBUS.main, values)


def create_lkas_ui_command(packer, main_on: bool, enabled: bool, steer_alert: bool, hud_control, stock_values: dict):
  """
  Creates a CAN message for the Ford IPC IPMA/LKAS status.

  Show the LKAS status with the "driver assist" lines in the IPC.

  Stock functionality is maintained by passing through unmodified signals.

  Frequency is 1Hz.
  """

  # LaActvStats_D_Dsply
  #    R  Intvn Warn Supprs Avail No
  # L
  # Intvn  24    19    14     9   4
  # Warn   23    18    13     8   3
  # Supprs 22    17    12     7   2
  # Avail  21    16    11     6   1
  # No     20    15    10     5   0
  #
  # TODO: test suppress state
  if enabled:
    lines = 0  # NoLeft_NoRight
    if hud_control.leftLaneDepart:
      lines += 4
    elif hud_control.leftLaneVisible:
      lines += 1
    if hud_control.rightLaneDepart:
      lines += 20
    elif hud_control.rightLaneVisible:
      lines += 5
  elif main_on:
    lines = 0
  else:
    if hud_control.leftLaneDepart:
      lines = 3  # WarnLeft_NoRight
    elif hud_control.rightLaneDepart:
      lines = 15  # NoLeft_WarnRight
    else:
      lines = 30  # LA_Off

  # TODO: use level 1 for no sound when less severe?
  hands_on_wheel_dsply = 2 if steer_alert else 0

  values = {
    **stock_values,
    "LaActvStats_D_Dsply": lines,                           # LKAS status (lines) [0|31]
    "LaHandsOff_D_Dsply": hands_on_wheel_dsply,             # 0=HandsOn, 1=Level1 (w/o chime), 2=Level2 (w/ chime), 3=Suppressed
  }
  return packer.make_can_msg("IPMA_Data", CANBUS.main, values)


def create_acc_ui_command(packer, main_on: bool, enabled: bool, hud_control, stock_values: dict):
  """
  Creates a CAN message for the Ford IPC adaptive cruise, forward collision warning and traffic jam
  assist status.

  Stock functionality is maintained by passing through unmodified signals.

  Frequency is 20Hz.
  """

  # Tja_D_Stat
  if enabled:
    if hud_control.leftLaneDepart:
      status = 3  # ActiveInterventionLeft
    elif hud_control.rightLaneDepart:
      status = 4  # ActiveInterventionRight
    else:
      status = 2  # Active
  elif main_on:
    if hud_control.leftLaneDepart:
      status = 5  # ActiveWarningLeft
    elif hud_control.rightLaneDepart:
      status = 6  # ActiveWarningRight
    else:
      status = 1  # Standby
  else:
    status = 0    # Off

  values = {
    **stock_values,
    "Tja_D_Stat": status,
  }
  return packer.make_can_msg("ACCDATA_3", CANBUS.main, values)


def create_button_command(packer, stock_values: dict, cancel = False, resume = False, tja_toggle = False, bus: int = CANBUS.camera):
  """
  Creates a CAN message for the Ford SCCM buttons/switches.

  Includes cruise control buttons, turn lights and more.
  """

  values = {
    **stock_values,
    "CcAslButtnCnclPress": 1 if cancel else 0,      # CC cancel button
    "CcAsllButtnResPress": 1 if resume else 0,      # CC resume button
    "TjaButtnOnOffPress": 1 if tja_toggle else 0,   # TJA toggle button
  }
  return packer.make_can_msg("Steering_Data_FD1", bus, values)
