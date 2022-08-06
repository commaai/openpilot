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


def create_lkas_ui_command(packer, main_on: bool, enabled: bool, steer_alert: bool, stock_values):
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
    "FeatConfigIpmaActl": stock_values["FeatConfigIpmaActl"],       # [0|65535]
    "FeatNoIpmaActl": stock_values["FeatNoIpmaActl"],               # [0|65535]
    "PersIndexIpma_D_Actl": stock_values["PersIndexIpma_D_Actl"],   # [0|7]
    "AhbcRampingV_D_Rq": stock_values["AhbcRampingV_D_Rq"],         # AHB ramping [0|3]
    "LaActvStats_D_Dsply": lines,                                   # LKAS status (lines) [0|31]
    "LaDenyStats_B_Dsply": stock_values["LaDenyStats_B_Dsply"],     # LKAS error [0|1]
    "LaHandsOff_D_Dsply": 2 if steer_alert else 0,                  # 0=HandsOn, 1=Level1 (w/o chime), 2=Level2 (w/ chime), 3=Suppressed
    "CamraDefog_B_Req": stock_values["CamraDefog_B_Req"],           # Windshield heater? [0|1]
    "CamraStats_D_Dsply": stock_values["CamraStats_D_Dsply"],       # Camera status [0|3]
    "DasAlrtLvl_D_Dsply": stock_values["DasAlrtLvl_D_Dsply"],       # DAS alert level [0|7]
    "DasStats_D_Dsply": stock_values["DasStats_D_Dsply"],           # DAS status [0|3]
    "DasWarn_D_Dsply": stock_values["DasWarn_D_Dsply"],             # DAS warning [0|3]
    "AhbHiBeam_D_Rq": stock_values["AhbHiBeam_D_Rq"],               # AHB status [0|3]
    "Set_Me_X1": stock_values["Set_Me_X1"],                         # [0|15]
  }
  return packer.make_can_msg("IPMA_Data", 0, values)


def create_acc_ui_command(packer, main_on: bool, enabled: bool, stock_values):
  """
  Creates a CAN message for the Ford IPC adaptive cruise, forward collision
  warning and traffic jam assist status.

  Stock functionality is maintained by passing through unmodified signals.

  Frequency is 20Hz.
  """

  values = {
    "HaDsply_No_Cs": stock_values["HaDsply_No_Cs"],                     # [0|255]
    "HaDsply_No_Cnt": stock_values["HaDsply_No_Cnt"],                   # [0|15]
    "AccStopStat_D_Dsply": stock_values["AccStopStat_D_Dsply"],         # ACC stopped status message: 0=NoDisplay, 1=ResumeReady, 2=Stopped, 3=PressResume [0|3]
    "AccTrgDist2_D_Dsply": stock_values["AccTrgDist2_D_Dsply"],         # ACC target distance [0|15]
    "AccStopRes_B_Dsply": stock_values["AccStopRes_B_Dsply"],           # [0|1]
    "TjaWarn_D_Rq": stock_values["TjaWarn_D_Rq"],                       # TJA warning: 0=NoWarning, 1=Cancel, 2=HardTakeOverLevel1, 3=HardTakeOverLevel2 [0|7]
    "Tja_D_Stat": 2 if enabled else (1 if main_on else 0),              # TJA status: 0=Off, 1=Standby, 2=Active, 3=InterventionLeft, 4=InterventionRight, 5=WarningLeft, 6=WarningRight, 7=NotUsed [0|7]
    "TjaMsgTxt_D_Dsply": stock_values["TjaMsgTxt_D_Dsply"],             # TJA text [0|7]
    "IaccLamp_D_Rq": stock_values["IaccLamp_D_Rq"],                     # iACC status icon [0|3]
    "AccMsgTxt_D2_Rq": stock_values["AccMsgTxt_D2_Rq"],                 # ACC text [0|15]
    "FcwDeny_B_Dsply": stock_values["FcwDeny_B_Dsply"],                 # FCW disabled [0|1]
    "FcwMemStat_B_Actl": stock_values["FcwMemStat_B_Actl"],             # FCW enabled setting [0|1]
    "AccTGap_B_Dsply": stock_values["AccTGap_B_Dsply"],                 # ACC time gap display setting [0|1]
    "CadsAlignIncplt_B_Actl": stock_values["CadsAlignIncplt_B_Actl"],   # Radar alignment? [0|1]
    "AccFllwMde_B_Dsply": stock_values["AccFllwMde_B_Dsply"],           # ACC follow mode display setting [0|1]
    "CadsRadrBlck_B_Actl": stock_values["CadsRadrBlck_B_Actl"],         # Radar blocked? [0|1]
    "CmbbPostEvnt_B_Dsply": stock_values["CmbbPostEvnt_B_Dsply"],       # AEB event status [0|1]
    "AccStopMde_B_Dsply": stock_values["AccStopMde_B_Dsply"],           # ACC stop mode display setting [0|1]
    "FcwMemSens_D_Actl": stock_values["FcwMemSens_D_Actl"],             # FCW sensitivity setting [0|3]
    "FcwMsgTxt_D_Rq": stock_values["FcwMsgTxt_D_Rq"],                   # FCW text [0|7]
    "AccWarn_D_Dsply": stock_values["AccWarn_D_Dsply"],                 # ACC warning [0|3]
    "FcwVisblWarn_B_Rq": stock_values["FcwVisblWarn_B_Rq"],             # FCW alert: 0=Off, 1=On [0|1]
    "FcwAudioWarn_B_Rq": stock_values["FcwAudioWarn_B_Rq"],             # FCW audio: 0=Off, 1=On [0|1]
    "AccTGap_D_Dsply": stock_values["AccTGap_D_Dsply"],                 # ACC time gap: 1=Time_Gap_1, 2=Time_Gap_2, ..., 5=Time_Gap_5 [0|7]
    "AccMemEnbl_B_RqDrv": stock_values["AccMemEnbl_B_RqDrv"],           # ACC setting: 0=NormalCruise, 1=AdaptiveCruise [0|1]
    "FdaMem_B_Stat": stock_values["FdaMem_B_Stat"],                     # FDA enabled setting [0|1]
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
