#!/usr/bin/env python3


# Copyright (C) 2016 Sixten Bergman
# License WTFPL
#
# This program is free software. It comes without any warranty, to the extent
# permitted by applicable law.
# You can redistribute it and/or modify it under the terms of the Do What The
# Fuck You Want To Public License, Version 2, as published by Sam Hocevar. See
# http://www.wtfpl.net/ for more details.
#
# note that the function peakdetect is derived from code which was released to
# public domain see: http://billauer.co.il/peakdet.html
#

from math import log
import numpy as np

__all__ = ["peakdetect"]



def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def _pad(fft_data, pad_len):
    """
    Pads fft data to interpolate in time domain

    keyword arguments:
    fft_data -- the fft
    pad_len --  By how many times the time resolution should be increased by

    return: padded list
    """
    length = len(fft_data)
    n = _n(length * pad_len)
    fft_data = list(fft_data)

    return fft_data[:length // 2] + [0] * (2**n-length) + fft_data[length // 2:]


def _n(x):
    """
    Find the smallest value for n, which fulfils 2**n >= x

    keyword arguments:
    x -- the value, which 2**n must surpass
      return: the integer n
    """
    return int(log(x)/log(2)) + 1


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html
      function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
      keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
      x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
      lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200)
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
      delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function
        return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   # Used to pop the first hit which almost always is false
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
      #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
              ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
        #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
    return [max_peaks, min_peaks]
