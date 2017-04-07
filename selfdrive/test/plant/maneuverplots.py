import os
import pickle
import sys

import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import pylab

from selfdrive.config import Conversions as CV

class ManeuverPlot(object):
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

    self.d_rel_array = []
    self.v_rel_array = []
    self.v_lead_array = []
    self.v_target_lead_array = []
    self.pid_speed_array = []
    self.cruise_speed_array = []
    self.jerk_factor_array = []

    self.a_target_min_array = []
    self.a_target_max_array = []

    self.v_target_array = []

    self.title = title
    
  def add_data(self, time, gas, brake, steer_torque, distance, speed, 
    acceleration, up_accel_cmd, ui_accel_cmd, d_rel, v_rel, v_lead, 
    v_target_lead, pid_speed, cruise_speed, jerk_factor, a_target_min, 
    a_target_max):
    self.time_array.append(time)
    self.gas_array.append(gas)
    self.brake_array.append(brake)
    self.steer_torque_array.append(steer_torque)
    self.distance_array.append(distance)
    self.speed_array.append(speed)
    self.acceleration_array.append(acceleration)
    self.up_accel_cmd_array.append(up_accel_cmd)
    self.ui_accel_cmd_array.append(ui_accel_cmd)
    self.d_rel_array.append(d_rel)
    self.v_rel_array.append(v_rel)
    self.v_lead_array.append(v_lead)
    self.v_target_lead_array.append(v_target_lead)
    self.pid_speed_array.append(pid_speed)
    self.cruise_speed_array.append(cruise_speed)
    self.jerk_factor_array.append(jerk_factor)
    self.a_target_min_array.append(a_target_min)
    self.a_target_max_array.append(a_target_max)

  def write_plot(self, path, maneuver_name):
    title = self.title or maneuver_name
    # TODO: Missing plots from the old one:
    # long_control_state
    # proportional_gb, intergral_gb
    maneuver_path = os.path.join(path, maneuver_name)
    if not os.path.exists(maneuver_path):
      os.makedirs(maneuver_path)

    with open(os.path.join(maneuver_path, "data.pickle"), "wb") as fout:
      pickle.dump(self, fout, protocol=-1)

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
    plt.plot(np.array(self.time_array), np.array(self.acceleration_array), 'g')
    self._plot_oscillated(np.array(self.time_array), np.array(self.a_target_min_array))
    self._plot_oscillated(np.array(self.time_array), np.array(self.a_target_max_array))
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.legend(['accel', 'max', 'min'], loc=0)
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
      np.array(self.time_array), np.array(self.ui_accel_cmd_array), 'b'
    )
    plt.xlabel("Time, [s]")
    plt.ylabel("Accel Cmd [m/s^2]")
    plt.grid()
    plt.legend(["Proportional", "Integral"], loc=0)
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
    
  @staticmethod
  def _detect_oscillations(arr, window=32, threshold=0.2, minstd=0.04):
    """
    Determines the intervals in `arr` which are unstable, that is, the values
    look like a sharp triangle wave. We apply the sliding window and calculate
    the decision statistic. That statistic is the number of times the
    piecewise-linear function generated from `arr` intersects the horizontal
    line at the height of the mean.

        /    /\      /\
      -/-\--/--\----/--    S = 5 (oscillating)
      /   \/    \__/
      
      -----------------    S = 0 (constant)
      
              /
           __/_            S = 1 (monotonical)
            /
           /
    
    @param arr The uniform time series data of type numpy.ndarray.
    
    @param window The size of the sliding window. The bigger the window, the
    more accurate is the detection but the poorer resolution.
          
    @param threshold Sets the sensitivity of the detection, the lower the value,
    the higher the sensitivity and the higher chance of getting false positives.
    
    @param minstd The minimum standard deviation value to allow the oscillation.
                  It eliminates trembling with low magnitude.
    
    @return list of found intervals, each interval is a [start, end) tuple.
    """
    threshold = int(window * threshold)
    detected = []
    start_pos = -1
    for i in range(len(arr) - window):
      crosses = np.diff(np.sign(arr[i:i + window] - np.mean(arr[i:i + window]))) \
        .astype(bool).sum()
      if crosses > threshold and np.std(arr[i:i + window]) > minstd:
        if start_pos < 0:
          start_pos = i
      elif start_pos >= 0:
        actual_start = start_pos
        # adjust for a constant prolog and epilog
        while arr[actual_start] == arr[actual_start + 1] and actual_start < len(arr) - 1:
          actual_start += 1
        actual_end = i + window
        while arr[actual_end] == arr[actual_end - 1] and actual_end > 0:
          actual_end -= 1
        # the following check is not strictly neccessary but it helps to filter
        # false positives
        if actual_end - actual_start > threshold:
          detected.append((actual_start, actual_end))
        start_pos = -1
    if start_pos >= 0:
      while arr[start_pos] == arr[start_pos + 1] and start_pos < len(arr) - 1:
        start_pos += 1
      detected.append((start_pos, len(arr)))
     
    if not detected:
      return []
    
    # postprocessing: merge intervals whih are near each other
    result = []
    start_pos, end_pos = detected[0]
    for next_start_pos, next_end_pos in detected[1:]:
      if next_start_pos - end_pos >= 2 * window:
        result.append((start_pos, end_pos))
        start_pos = next_start_pos
      end_pos = next_end_pos
    result.append((start_pos, end_pos))
    return result
    
  @staticmethod
  def _estimate_oscillation_properties(arr, window=64):
    """
    Provides the smoothed average trend, the maximum and minimum borders on
    an oscillation interval found by _detect_oscillations().
    Minimum and maximum borders are claculated as the linear interpolation of
    the peaks. We calculate the moving average on the window after augmenting
    the data at the edges using the mean values within half of the window size.
    
    @param arr The uniform time series data on the oscillation interval of type
               numpy.ndarray.
               
    @param window The smoothing window size. The bigger the value, the smoother
                  the result but the less details on the peaks.
    @return [smoothed average, mins, maxs], each is numpy.ndarray of the same
            shape as `arr`
    """
    borders = []
    length = len(arr)
    for op in (np.less_equal, np.greater_equal):
      iextrs = argrelextrema(arr, op, order=window // 2)[0]
      if len(iextrs):
        borders.append(np.interp(range(length), iextrs, arr[iextrs]))
      else:
        # well, this is strange
        raise ValueError("bug in ManeuverPlot._detect_oscillations()")
    arr = np.pad(arr, window // 2, 'mean', stat_length=window // 2)
    moving_avg = np.convolve(
      arr, np.ones((window,)) / window, mode='valid')[:length]
    return [moving_avg] + borders
    
  @classmethod
  def _plot_oscillated(
      cls, time_array, series_array, detection_window=32, averaging_window=64,
      style='k--', avg_style=None, min_style=None, max_style=None,
      min_color="blue", max_color="red"):
    """
    Plots the time series with the special handling of the intervals with
    oscillating values.
    
    @param time_array numpy.ndarray with the time measurements.
    @param series_array numpy.ndarray with the corresponding value measurements.
    @param detection_window Window size for oscillation detection (see
                            _detect_oscillations()).
    @param averaging_window Window size for the smoothing of the oscillating
                            values (see _estimate_oscillation_properties()).
    @param style MPL style for the regular parts of the plot.
    @param avg_style MPL style of the smoothed oscillations.
    @param min_style MPL style of the minimum borders of the oscillations.
    @param max_style MPL style of the maximum borders of the oscillations.
    @param min_color MPL color of the minimum borders of the oscillations.
    @param max_color MPL color of the maximum borders of the oscillations.
    """
    oscillations = cls._detect_oscillations(series_array, window=detection_window)
    previous_pos = 0
    for interval in oscillations:
      try:
        avg, mins, maxs = cls._estimate_oscillation_properties(
            series_array[interval[0]:interval[1]], window=averaging_window)
      except ValueError:
        # we failed to detect oscillations correctly
        sys.stderr.write("plot: failed to handle the oscillations\n")
        plt.plot(time_array[previous_pos:interval[1]],
                 series_array[previous_pos:interval[1]],
                 style, alpha=0.5)
        previous_pos = interval[1]
        continue

      plt.plot(time_array[previous_pos:interval[0]],
               series_array[previous_pos:interval[0]],
               style)
      # make sure we have a connected line
      avg[0] = series_array[interval[0] - 1]
      if interval[1] < len(series_array):
        avg[-1] = series_array[interval[1]]
      plt.plot(time_array[interval[0]:interval[1]], avg,
               avg_style if avg_style else style)
      plt.plot(time_array[interval[0]:interval[1]], mins,
               min_style if min_style else style, color=min_color)
      plt.plot(time_array[interval[0]:interval[1]], maxs,
               max_style if max_style else style, color=max_color)
      previous_pos = interval[1]
    plt.plot(time_array[previous_pos:], series_array[previous_pos:], style)

