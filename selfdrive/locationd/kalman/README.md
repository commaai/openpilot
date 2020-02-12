# Kalman filter library

## Introduction
The kalman filter framework described here is an incredibly powerful tool for any optimization problem,
but particularly for visual odometry, sensor fusion localization or SLAM. It is designed to provide the very
accurate results, work online or offline, be fairly computationally efficient, be easy to design filters with in
python.

## Feature walkthrough

### Extended Kalman Filter with symbolic Jacobian computation
Most dynamic systems can be described as a Hidden Markov Process. To estimate the state of such a system with noisy
measurements one can use a Recursive Bayesian estimator. For a linear Markov Process a regular linear Kalman filter is optimal.
Unfortunately, a lot of systems are non-linear. Extended Kalman Filters can model systems by linearizing the non-linear
system at every step, this provides a close to optimal estimator when the linearization is good enough. If the linearization
introduces too much noise, one can use an Iterated Extended Kalman Filter, Unscented Kalman Filter or a Particle Filter. For
most applications those estimators are overkill and introduce too much complexity and require a lot of additional compute.

Conventionally Extended Kalman Filters are implemented by writing the system's dynamic equations and then manually symbolically 
calculating the Jacobians for the linearization. For complex systems this is time consuming and very prone to calculation errors.
This library symbolically computes the Jacobians using sympy to remove the possiblity of introducing calculation errors.

### Error State Kalman Filter
3D localization algorithms ussually also require estimating orientation of an object in 3D. Orientation is generally represented
with euler angles or quaternions. 

Euler angles have several problems, there are mulitple ways to represent the same orientation,
gimbal lock can cause the loss of a degree of freedom and lastly their behaviour is very non-linear when errors are large. 
Quaternions with one strictly positive dimension don't suffer from these issues, but have another set of problems.
Quaternions need to be normalized otherwise they will grow unbounded, this is cannot be cleanly enforced in a kalman filter.
Most importantly though a quaternion has 4 dimensions, but only represents 3 DOF. It is problematic to describe the error-state,
with redundant dimensions.

The solution is to have a compromise, use the quaternion to represent the system's state and use euler angles to describe
the error-state. This library supports and defining an arbitrary error-state that is different from the state.

### Multi-State Constraint Kalman Filter
paper:

### Rauch–Tung–Striebel smoothing
When doing offline estimation with a kalman filter there can be an initialzition period where states are badly estimated. 
Global estimators don't suffer from this, to make our kalman filter competitive with global optimizers when can run the filter
backwards using an RTS smoother. Those combined with potentially multiple forward and backwards passes of the data should make
performance very close to global optimization.

### Mahalanobis distance outlier rejector
A lot of measurements do not come from a Gaussian distribution and as such have outliers that do not fit the statistical model
of the Kalman filter. This can cause a lot of performance issues if not dealt with. This library allows the use of a mahalanobis
distance statistical test on the incoming measurements to deal with this. Note that good initialization is critical to prevent
good measurements from being rejected.


