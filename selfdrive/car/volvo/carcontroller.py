from common.numpy_fast import clip, interp
from selfdrive.car.volvo.values import CAR, PLATFORM, DBC, CarControllerParams as CCP 
from selfdrive.car.volvo import volvocan
from opendbc.can.packer import CANPacker
from collections import deque

class SteerCommand:
  angle_request = 0
  steer_direction = 0
  trqlim = 0


class CarController():
  def __init__(self, dbc_name, CP, VW):
    
    # state
    self.acc_enabled_prev = 0

    # steering related
    self.angle_request_prev = 0
    
    # Direction change statemachine
    self.UNBLOCKED = 0
    self.BLOCKED = 1
    self.BLOCK_LEN = CCP.BLOCK_LEN  # Block steer direction change for x samples

    self.dir_state = 0
    self.block_steering = 0
    self.steer_direction_bf_block = 0
    self.des_steer_direction_prev = 0

    # SteerCommand
    self.SteerCommand = SteerCommand
    self.trq_fifo = deque([])  
    self.fault_frame = -200

    # Diag
    self.doDTCRequests = True  # Turn on and off DTC requests
    self.checkPN = False       # Check partnumbers
    self.clearDtcs = False     # Set false to stop sending diagnostic requests 
    self.timeout = 0           # Set to 0 as init
    self.diagRequest = { 
      "byte0": 0x03,
      "byte1": 0x19,
      "byte2": 0x02,
      "byte3": 0x02,
      }
    self.flowControl = { 
      "byte0": 0x30,
      "byte1": 0x00,
      "byte2": 0x00,
      "byte3": 0x00,
      }
    self.clearDTC = {
      "byte0": 0x04,
      "byte1": 0x14,
      "byte2": 0xFF,
      "byte3": 0xFF,
      "byte4": 0xFF,
      }

    # Part number
    self.cnt = 0          # Init at 0 always
    self.sndNxtFrame = 0  # Init at low value 
    self.dictKeys = ["byte"+str(x) for x in range(8)]
    startdid = 0xf1a1     # Start with this DID (Data IDentifier, read UDS Spec for more info)
    self.dids = [x for x in range(startdid, startdid+9)]

    # Setup detection helper. Routes commands to
    # an appropriate CAN bus number.
    self.CP = CP
    self.packer = CANPacker(DBC[CP.carFingerprint]['pt'])


  def max_angle_req(self, current_steer_angle, angle_request_prev, CCP):
    """ 
    Calculate maximum angle request delta/offset from current steering angle. 
    
    This is just a helper function that calculates the boundary for min and max
    steering angle request. It uses the parameters CCP.MAX_ACT_ANGLE_REQUEST_DIFF
    and CCP.STEER_ANGLE_DELTA_REQ_DIFF. To calculate the max and min allowed delta/offset request.

    The delta request is just a rate limiter. The request angle cant change more 
    than CCP.STEER_ANGLE_DELTA_REQ_DIFF per loop. 
    
    """

    # determine max and min allowed lka angle request
    # based on delta per sample
    max_delta_right = angle_request_prev-CCP.STEER_ANGLE_DELTA_REQ_DIFF 
    max_delta_left = angle_request_prev+CCP.STEER_ANGLE_DELTA_REQ_DIFF

    # based on distance from actual steering angle
    max_right = current_steer_angle-CCP.MAX_ACT_ANGLE_REQUEST_DIFF 
    max_left = current_steer_angle+CCP.MAX_ACT_ANGLE_REQUEST_DIFF

    return max_right, max_left, max_delta_right, max_delta_left

  def dir_change(self, steer_direction, error):
    """ Filters out direction changes
    
    Uses a simple state machine to determine if we should 
    block or allow the steer_direction bits to pass thru.

    """
    
    dessd = steer_direction
    dzError = 0 if abs(error) < CCP.DEADZONE else error 
    tState = -1 

    # Update prev with desired if just enabled.
    self.des_steer_direction_prev = steer_direction if not self.acc_enabled_prev else self.des_steer_direction_prev
    
    # Check conditions for state change
    if self.dir_state == self.UNBLOCKED:
      tState = self.BLOCKED if (steer_direction != self.des_steer_direction_prev and dzError != 0) else tState
    elif self.dir_state == self.BLOCKED:
      if (steer_direction == self.steer_direction_bf_block) or (self.block_steering <= 0) or (dzError == 0):
        tState = self.UNBLOCKED

    # State transition
    if tState == self.UNBLOCKED:
      self.dir_state = self.UNBLOCKED
    elif tState == self.BLOCKED:
      self.steer_direction_bf_block = self.des_steer_direction_prev  
      self.block_steering = self.BLOCK_LEN
      self.dir_state = self.BLOCKED

    #  Run actions in state
    if self.dir_state == self.UNBLOCKED:
      if dzError == 0:
        steer_direction = self.des_steer_direction_prev # Set old request when inside deadzone
    if self.dir_state == self.BLOCKED:
      self.block_steering -= 1
      steer_direction = CCP.STEER_NO

    #print("State:{} Sd:{} Sdp:{} Bs:{} Dz:{:.2f} Err:{:.2f}".format(self.dir_state, steer_direction, self.des_steer_direction_prev, self.block_steering, dzError, error))
    return steer_direction

  def update(self, enabled, CS, frame,
             actuators, 
             visualAlert, leftLaneVisible,
             rightLaneVisible, leadVisible,
             leftLaneDepart, rightLaneDepart):
    """ Controls thread """
    
    # Send CAN commands.
    can_sends = []

    # run at 50hz
    if (frame % 2 == 0):
      fingerprint = self.CP.carFingerprint
      
      if enabled and CS.out.vEgo > self.CP.minSteerSpeed:
        current_steer_angle = CS.out.steeringAngle
        self.SteerCommand.angle_request = actuators.steerAngle # Desired value from pathplanner
        
        # # windup slower
        if self.angle_request_prev * self.SteerCommand.angle_request > 0. and abs(self.SteerCommand.angle_request) > abs(self.angle_request_prev):
          angle_rate_lim = interp(CS.out.vEgo, CCP.ANGLE_DELTA_BP, CCP.ANGLE_DELTA_V)
        else:
          angle_rate_lim = interp(CS.out.vEgo, CCP.ANGLE_DELTA_BP, CCP.ANGLE_DELTA_VU)

        self.SteerCommand.angle_request = clip(self.SteerCommand.angle_request, self.angle_request_prev - angle_rate_lim, self.angle_request_prev + angle_rate_lim)

        # Create trqlim from angle request (before constraints)
        if fingerprint in PLATFORM.C1:
          self.SteerCommand.trqlim = -127 if current_steer_angle > self.SteerCommand.angle_request else 127
          self.SteerCommand.steer_direction = CCP.STEER
        elif fingerprint in PLATFORM.EUCD:
          self.SteerCommand.trqlim = 0
          # MIGHT be needed for EUCD
          self.SteerCommand.steer_direction = CCP.STEER_RIGHT if current_steer_angle > self.SteerCommand.angle_request else CCP.STEER_LEFT
          self.SteerCommand.steer_direction = self.dir_change(self.SteerCommand.steer_direction, current_steer_angle-self.SteerCommand.angle_request) # Filter the direction change 
          
      else:
        self.SteerCommand.steer_direction = CCP.STEER_NO
        self.SteerCommand.trqlim = 0
        if fingerprint in PLATFORM.C1:
          self.SteerCommand.angle_request = clip(CS.out.steeringAngle, -359.95, 359.90)  # Cap values at max min values (Cap 2 steps from max min). Max=359.99445, Min=-360.0384 
        else:
          self.SteerCommand.angle_request = 0

      
      # Count no of consequtive samples of zero torque by lka.
      # Try to recover, blocking steering request for 2 seconds.
      if fingerprint in PLATFORM.C1:
        if enabled and CS.out.vEgo > self.CP.minSteerSpeed:
          self.trq_fifo.append(CS.PSCMInfo.LKATorque)
          if len(self.trq_fifo) > CCP.N_ZERO_TRQ:
            self.trq_fifo.popleft()
        else:
          self.trq_fifo.clear()
          self.fault_frame = -200

        if (self.trq_fifo.count(0) >= CCP.N_ZERO_TRQ) and (self.fault_frame == -200):
          self.fault_frame = frame+100

        if enabled and (frame < self.fault_frame):
          self.SteerCommand.steer_direction = CCP.STEER_NO

        if frame > self.fault_frame+8:  # Ignore steerWarning for another 8 samples.
          self.fault_frame = -200     


      # update stored values
      self.acc_enabled_prev = enabled
      self.angle_request_prev = self.SteerCommand.angle_request
      if self.SteerCommand.steer_direction == CCP.STEER_RIGHT or self.SteerCommand.steer_direction == CCP.STEER_LEFT: # TODO: Move this inside dir_change, think it should work?
        self.des_steer_direction_prev = self.SteerCommand.steer_direction  # Used for dir_change function
      
      # Manipulate data from servo to FSM
      # Avoid fault codes, that will stop LKA
      can_sends.append(volvocan.manipulateServo(self.packer, self.CP.carFingerprint, CS))
    
      # send can, add to list.
      can_sends.append(volvocan.create_steering_control(self.packer, frame, self.CP.carFingerprint, self.SteerCommand, CS.FSMInfo))

    
    # Cancel ACC if engaged when OP is not.
    if not enabled and CS.out.cruiseState.enabled:
      can_sends.append(volvocan.cancelACC(self.packer, self.CP.carFingerprint, CS))


    # Send diagnostic requests
    if(self.doDTCRequests):
      if(frame % 100 == 0) and (not self.clearDtcs):
        # Request diagnostic codes, 2 Hz
        can_sends.append(self.packer.make_can_msg("diagFSMReq", 2, self.diagRequest))
        #can_sends.append(self.packer.make_can_msg("diagGlobalReq", 2, self.diagRequest))
        can_sends.append(self.packer.make_can_msg("diagGlobalReq", 0, self.diagRequest))
        #can_sends.append(self.packer.make_can_msg("diagPSCMReq", 0, self.diagRequest))
        #can_sends.append(self.packer.make_can_msg("diagCEMReq", 0, self.diagRequest))
        #can_sends.append(self.packer.make_can_msg("diagCVMReq", 0, self.diagRequest))
        self.timeout = frame + 5 # Set wait time 
      
      # Handle flow control in case of many DTC
      if frame > self.timeout and self.timeout > 0: # Wait fix time before sending flow control, otherwise just spamming...
        self.timeout = 0 
        if (CS.diag.diagFSMResp & 0x10000000):
          can_sends.append(self.packer.make_can_msg("diagFSMReq", 2, self.flowControl))
        if (CS.diag.diagCEMResp & 0x10000000):
          can_sends.append(self.packer.make_can_msg("diagCEMReq", 0, self.flowControl))
        if (CS.diag.diagPSCMResp & 0x10000000):
          can_sends.append(self.packer.make_can_msg("diagPSCMReq", 0, self.flowControl))
        if (CS.diag.diagCVMResp & 0x10000000):
          can_sends.append(self.packer.make_can_msg("diagCVMReq", 0, self.flowControl))

      # Check part numbers
      if self.checkPN and frame > 100 and frame > self.sndNxtFrame:
        if self.cnt < len(self.dids):
          did = [0x03, 0x22, (self.dids[self.cnt] & 0xff00)>>8, self.dids[self.cnt] & 0x00ff] # Create diagnostic command
          did.extend([0]*(8-len(did))) 
          diagReq = dict(zip(self.dictKeys,did))
          #can_sends.append(self.packer.make_can_msg("diagGlobalReq", 2, diagReq))
          #can_sends.append(self.packer.make_can_msg("diagGlobalReq", 0, diagReq))
          can_sends.append(self.packer.make_can_msg("diagFSMReq", 2, diagReq))
          can_sends.append(self.packer.make_can_msg("diagCEMReq", 0, diagReq))
          can_sends.append(self.packer.make_can_msg("diagPSCMReq", 0, diagReq))
          can_sends.append(self.packer.make_can_msg("diagCVMReq", 0, diagReq))
          self.cnt += 1
          self.timeout = frame+5             # When to send flowControl
          self.sndNxtFrame = self.timeout+5  # When to send next part number request

        elif True:                           # Stop when list has been looped thru.
          self.checkPN = False

      # Clear DTCs in FSM on start
      # TODO check for engine running before clearing dtc.
      if(self.clearDtcs and (frame > 0) and (frame % 500 == 0)):
        can_sends.append(self.packer.make_can_msg("diagGlobalReq", 0, self.clearDTC))
        can_sends.append(self.packer.make_can_msg("diagFSMReq", 2, self.clearDTC))
        #can_sends.append(self.packer.make_can_msg("diagPSCMReq", 0, self.clearDTC))
        #can_sends.append(self.packer.make_can_msg("diagCEMReq", 0, self.clearDTC))
        self.clearDtcs = False
      
    return can_sends
