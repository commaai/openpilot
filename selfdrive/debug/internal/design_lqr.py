#!/usr/bin/env python3
import numpy as np
import control

dt = 0.01
A = np.array([[ 0.        ,  1.        ],       [-0.78823806,  1.78060701]])
B = np.array([[-2.23399437e-05],       [ 7.58330763e-08]])
C = np.array([[1., 0.]])


# Kalman tuning
Q = np.diag([1, 1])
R = np.atleast_2d(1e5)

(_, _, L) = control.dare(A.T, C.T, Q, R)
L = L.T

# LQR tuning
Q = np.diag([2e5, 1e-5])
R = np.atleast_2d(1)
(_, _, K) = control.dare(A, B, Q, R)

A_cl = (A - B.dot(K))
sys = control.ss(A_cl, B, C, 0, dt)
dc_gain = control.dcgain(sys)

print(("self.A = np." + A.__repr__()).replace('\n', ''))
print(("self.B = np." + B.__repr__()).replace('\n', ''))
print(("self.C = np." + C.__repr__()).replace('\n', ''))
print(("self.K = np." + K.__repr__()).replace('\n', ''))
print(("self.L = np." + L.__repr__()).replace('\n', ''))
print("self.dc_gain = " + str(dc_gain))
