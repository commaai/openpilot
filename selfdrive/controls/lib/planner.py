#!/usr/bin/env python
import zmq
from common.realtime import sec_since_boot
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.pathplanner import PathPlanner
from selfdrive.controls.lib.adaptivecruise import AdaptiveCruise
from selfdrive.controls.lib.fcw import ForwardCollisionWarning

_DT = 0.01   # 100Hz


class Planner(object):
  def __init__(self, CP):
    context = zmq.Context()
    self.CP = CP
    self.live20 = messaging.sub_sock(context, service_list['live20'].port)
    self.model = messaging.sub_sock(context, service_list['model'].port)

    self.plan = messaging.pub_sock(context, service_list['plan'].port)

    self.last_md_ts = 0
    self.last_l20_ts = 0
    self.last_model = 0.
    self.last_l20 = 0.
    self.model_dead = True
    self.radar_dead = True
    self.radar_errors = []

    self.PP = PathPlanner()
    self.AC = AdaptiveCruise()
    self.FCW = ForwardCollisionWarning(_DT)

  # this runs whenever we get a packet that can change the plan
  def update(self, CS, LoC):
    cur_time = sec_since_boot()

    md = messaging.recv_sock(self.model)
    if md is not None:
      self.last_md_ts = md.logMonoTime
      self.last_model = cur_time
      self.model_dead = False
    if cur_time - self.last_model > 0.5:
      self.model_dead = True

    l20 = messaging.recv_sock(self.live20)
    if l20 is not None:
      self.last_l20_ts = l20.logMonoTime
      self.last_l20 = cur_time
      self.radar_dead = False
      self.radar_errors = list(l20.live20.radarErrors)
    if cur_time - self.last_l20 > 0.5:
      self.radar_dead = True

    self.PP.update(CS.vEgo, md)

    # LoC.v_pid -> CS.vEgo
    # TODO: is this change okay?
    self.AC.update(CS.vEgo, CS.steeringAngle, LoC.v_pid, self.CP, l20)

    # **** send the plan ****
    plan_send = messaging.new_message()
    plan_send.init('plan')

    events = []
    if self.model_dead:
      events.append(create_event('modelCommIssue', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if self.radar_dead or 'commIssue' in self.radar_errors:
      events.append(create_event('radarCommIssue', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if 'fault' in self.radar_errors:
      events.append(create_event('radarFault', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

    plan_send.plan.events = events

    plan_send.plan.mdMonoTime = self.last_md_ts
    plan_send.plan.l20MonoTime = self.last_l20_ts

    # lateral plan
    plan_send.plan.lateralValid = not self.model_dead
    plan_send.plan.dPoly = map(float, self.PP.d_poly)
    plan_send.plan.laneWidth = float(self.PP.lane_width)

    # longitudal plan
    plan_send.plan.longitudinalValid = not self.radar_dead
    plan_send.plan.vTarget = float(self.AC.v_target_lead)
    plan_send.plan.aTargetMin = float(self.AC.a_target[0])
    plan_send.plan.aTargetMax = float(self.AC.a_target[1])
    plan_send.plan.jerkFactor = float(self.AC.jerk_factor)
    plan_send.plan.hasLead = self.AC.has_lead

    # compute risk of collision events: fcw
    self.FCW.process(CS, self.AC)
    plan_send.plan.fcw = bool(self.FCW.active)

    self.plan.send(plan_send.to_bytes())
    return plan_send
