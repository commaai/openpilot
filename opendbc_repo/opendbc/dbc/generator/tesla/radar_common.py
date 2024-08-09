#!/usr/bin/env python3

def get_radar_point_definition(base_id, base_name):
  return f"""
BO_ {base_id} {base_name}_A: 8 Radar
 SG_ LongDist : 0|12@1+ (0.0625,0) [0|255.9] "meters"  Autopilot
 SG_ LongSpeed : 12|12@1+ (0.0625,-128) [-128|128] "meters/sec"  Autopilot
 SG_ LatDist : 24|11@1+ (0.125,-128) [-128|128] "meters"  Autopilot
 SG_ ProbExist : 35|5@1+ (3.125,0) [0|96.875] "%"  Autopilot
 SG_ LongAccel : 40|10@1+ (0.03125,-16) [-16|16] "meters/sec/sec"  Autopilot
 SG_ ProbObstacle : 50|5@1+ (3.125,0) [0|96.875] "%"  Autopilot
 SG_ Valid : 55|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ ProbNonObstacle : 56|5@1+ (3.125,0) [0|96.875] "%"  Autopilot
 SG_ Meas : 61|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ Tracked : 62|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ Index : 63|1@1+ (1,0) [0|1] ""  Autopilot

BO_ {base_id+1} {base_name}_B: 8 Radar
 SG_ LatSpeed : 0|10@1+ (0.125,-64) [-64|64] "meters/sec"  Autopilot
 SG_ Length : 10|6@1+ (0.125,0) [0|7.875] "m"  Autopilot
 SG_ dZ : 16|6@1+ (0.25,-5) [-5|10.75] "m"  Autopilot
 SG_ MovingState : 22|2@1+ (1,0) [0|3] ""  Autopilot
 SG_ dxSigma : 24|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ vxSigma : 30|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ axSigma : 36|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ dySigma : 42|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ ProbClass : 48|5@1+ (3.125,0) [0|96.875] "%"  Autopilot
 SG_ Class : 53|3@1+ (1,0) [0|7] ""  Autopilot
 SG_ dxRearEndLoss : 56|6@1+ (1,0) [0|63] ""  Autopilot
 SG_ NotUsed : 62|1@1+ (1,0) [0|1] ""  Autopilot
 SG_ Index2 : 63|1@1+ (1,0) [0|1] ""  Autopilot
"""

def get_val_definition(base_id):
  return f"""
VAL_ {base_id+1} MovingState 3 "RADAR_MOVESTATE_STANDING" 2 "RADAR_MOVESTATE_STOPPED" 1 "RADAR_MOVESTATE_MOVING" 0 "RADAR_MOVESTATE_INDETERMINATE" ;
VAL_ {base_id+1} Class 4 "RADAR_CLASS_CONSTRUCTION_ELEMENT" 3 "RADAR_CLASS_MOVING_PEDESTRIAN" 2 "RADAR_CLASS_MOVING_TWO_WHEEL_VEHICLE" 1 \
"RADAR_CLASS_MOVING_FOUR_WHEEL_VEHICLE" 0 "RADAR_CLASS_UNKNOWN" ;"""