import os

import numpy as np
import matplotlib.pyplot as plt
import pylab

from selfdrive.config import Conversions as CV

class ManeuverPlot():
  def __init__(self, title = None):
    self.time_array = []

    self.gas_array =  []
    self.brake_array = []
    self.steer_torque_array = []

    self.distance_array = []
    self.speed_array = []
    self.acceleration_array = []

    self.up_accel_cmd_array = []
    self.ui_accel_cmd_array = []
    self.uf_accel_cmd_array = []

    self.d_rel_array = []
    self.v_rel_array = []
    self.v_lead_array = []
    self.v_target_lead_array = []
    self.pid_speed_array = []
    self.cruise_speed_array = []
    self.jerk_factor_array = []

    self.a_target_array = []

    self.v_target_array = []

    self.fcw_array = []

    self.title = title
    
  def add_data(self, time, gas, brake, steer_torque, distance, speed, 
    acceleration, up_accel_cmd, ui_accel_cmd, uf_accel_cmd, d_rel, v_rel, 
    v_lead, v_target_lead, pid_speed, cruise_speed, jerk_factor, a_target, fcw):
    self.time_array.append(time)
    self.gas_array.append(gas)
    self.brake_array.append(brake)
    self.steer_torque_array.append(steer_torque)
    self.distance_array.append(distance)
    self.speed_array.append(speed)
    self.acceleration_array.append(acceleration)
    self.up_accel_cmd_array.append(up_accel_cmd)
    self.ui_accel_cmd_array.append(ui_accel_cmd)
    self.uf_accel_cmd_array.append(uf_accel_cmd)
    self.d_rel_array.append(d_rel)
    self.v_rel_array.append(v_rel)
    self.v_lead_array.append(v_lead)
    self.v_target_lead_array.append(v_target_lead)
    self.pid_speed_array.append(pid_speed)
    self.cruise_speed_array.append(cruise_speed)
    self.jerk_factor_array.append(jerk_factor)
    self.a_target_array.append(a_target)
    self.fcw_array.append(fcw)


  def write_plot(self, path, maneuver_name):
    # title = self.title or maneuver_name
    # TODO: Missing plots from the old one:
    # long_control_state
    # proportional_gb, intergral_gb
    if not os.path.exists(path + "/" + maneuver_name):
      os.makedirs(path + "/" + maneuver_name)
    plt_num = 0
    
    # speed chart ===================
    plt_num += 1
    plt.figure(plt_num)
    plt.plot(
      np.array(self.time_array), np.array(self.speed_array) * CV.MS_TO_MPH, 'r',
      np.array(self.time_array), np.array(self.pid_speed_array) * CV.MS_TO_MPH, 'y--',
      np.array(self.time_array), np.array(self.v_target_lead_array) * CV.MS_TO_MPH, 'b',
      np.array(self.time_array), np.array(self.cruise_speed_array) * CV.KPH_TO_MPH, 'k',
      np.array(self.time_array), np.array(self.v_lead_array) * CV.MS_TO_MPH, 'm'
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [mph]')
    plt.legend(['speed', 'pid speed', 'Target (lead) speed', 'Cruise speed', 'Lead speed'], loc=0)
    plt.grid()
    pylab.savefig("/".join([path, maneuver_name, 'speeds.svg']), dpi=1000)

    # acceleration chart ============
    plt_num += 1
    plt.figure(plt_num)
    plt.plot(
      np.array(self.time_array), np.array(self.acceleration_array), 'g',
      np.array(self.time_array), np.array(self.a_target_array), 'k--',
      np.array(self.time_array), np.array(self.fcw_array), 'ro',
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.legend(['ego-plant', 'target', 'fcw'], loc=0)
    plt.grid()
    pylab.savefig("/".join([path, maneuver_name, 'acceleration.svg']), dpi=1000)

    # pedal chart ===================
    plt_num += 1
    plt.figure(plt_num)
    plt.plot(
      np.array(self.time_array), np.array(self.gas_array), 'g',
      np.array(self.time_array), np.array(self.brake_array), 'r',
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Pedal []')
    plt.legend(['Gas pedal', 'Brake pedal'], loc=0)
    plt.grid()
    pylab.savefig("/".join([path, maneuver_name, 'pedals.svg']), dpi=1000)

    # pid chart ======================
    plt_num += 1
    plt.figure(plt_num)
    plt.plot(
      np.array(self.time_array), np.array(self.up_accel_cmd_array), 'g',
      np.array(self.time_array), np.array(self.ui_accel_cmd_array), 'b',
      np.array(self.time_array), np.array(self.uf_accel_cmd_array), 'r'
    )
    plt.xlabel("Time, [s]")
    plt.ylabel("Accel Cmd [m/s^2]")
    plt.grid()
    plt.legend(["Proportional", "Integral", "feedforward"], loc=0)
    pylab.savefig("/".join([path, maneuver_name, "pid.svg"]), dpi=1000)

    # relative distances chart =======
    plt_num += 1
    plt.figure(plt_num)
    plt.plot(
      np.array(self.time_array), np.array(self.d_rel_array), 'g',
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Relative Distance [m]')
    plt.grid()
    pylab.savefig("/".join([path, maneuver_name, 'distance.svg']), dpi=1000)

    plt.close("all")

