import abc
import numpy as np
import numpy.matlib
# The EKF class contains the framework for an Extended Kalman Filter, but must be subclassed to use.
# A subclass must implement:
#   1) calc_transfer_fun(); see bottom of file for more info.
#   2) __init__() to initialize self.state, self.covar, and self.process_noise appropriately

# Alternatively, the existing implementations of EKF can be used (e.g. EKF2D)

# Sensor classes are optionally used to pass measurement information into the EKF, to keep
#   sensor parameters and processing methods for a each sensor together.
# Sensor classes have a read() method which takes raw sensor data and returns
#   a SensorReading object, which can be passed to the EKF update() method.

# For usage, see run_ekf1d.py in selfdrive/new for a simple example.
# ekf.predict(dt) should be called between update cycles with the time since it was last called.
# Ideally, predict(dt) should be called at a relatively constant rate.
# update() should be called once per sensor, and can be called multiple times between predict steps.
# Access and set the state of the filter directly with ekf.state and ekf.covar.

class SensorReading:
  # Given a perfect model and no noise, data = obs_model * state
  def __init__(self, data, covar, obs_model):
    self.data = data
    self.obs_model = obs_model
    self.covar = covar

  def __repr__(self):
    return "SensorReading(data={}, covar={}, obs_model={})".format(
      repr(self.data), repr(self.covar), repr(self.obs_model))


# A generic sensor class that does no pre-processing of data
class SimpleSensor:
  # obs_model can be 
  #   a full obesrvation model matrix, or
  #   an integer or tuple of indices into ekf.state, indicating which variables are being directly observed
  # covar can be
  #   a full covariance matrix
  #   a float or tuple of individual covars for each component of the sensor reading
  # dims is the number of states in the EKF
  def __init__(self, obs_model, covar, dims):
    # Allow for integer covar/obs_model
    if not hasattr(obs_model, "__len__"):
      obs_model = (obs_model, )
    if not hasattr(covar, "__len__"):
      covar = (covar, )

    # Full observation model passed
    if dims in np.array(obs_model).shape:
      self.obs_model = np.asmatrix(obs_model)
      self.covar = np.asmatrix(covar)
    # Indices of unit observations passed
    else:
      self.obs_model = np.matlib.zeros((len(obs_model), dims))
      self.obs_model[:, list(obs_model)] = np.identity(len(obs_model))
      if np.asarray(covar).ndim == 2:
        self.covar = np.asmatrix(covar)
      elif len(covar) == len(obs_model):
        self.covar = np.matlib.diag(covar)
      else:
        self.covar = np.matlib.identity(len(obs_model)) * covar

  def read(self, data, covar=None):
    if covar:
      self.covar = covar
    return SensorReading(data, self.covar, self.obs_model)


class EKF:
  __metaclass__ = abc.ABCMeta

  def __init__(self, debug=False):
    self.DEBUG = debug


  def __str__(self):
    return "EKF(state={}, covar={})".format(self.state, self.covar)

  # Measurement update
  # Reading should be a SensorReading object with data, covar, and obs_model attributes
  def update(self, reading):
    # Potential improvements:
    # deal with negative covars
    # add noise to really low covars to ensure stability
    # use mahalanobis distance to reject outliers
    # wrap angles after state updates and innovation

    # y = z - H*x
    innovation = reading.data - reading.obs_model * self.state

    if self.DEBUG:
      print "reading:\n",reading.data
      print "innovation:\n",innovation

    # S = H*P*H' + R
    innovation_covar = reading.obs_model * self.covar * reading.obs_model.T + reading.covar

    # K = P*H'*S^-1
    kalman_gain = self.covar * reading.obs_model.T * np.linalg.inv(
      innovation_covar)

    if self.DEBUG:
      print "gain:\n", kalman_gain
      print "innovation_covar:\n", innovation_covar
      print "innovation: ", innovation
      print "test: ", self.covar * reading.obs_model.T * (
        reading.obs_model * self.covar * reading.obs_model.T + reading.covar *
        0).I

    # x = x + K*y
    self.state += kalman_gain*innovation

    # print "covar", np.diag(self.covar)
    #self.state[(roll_vel, yaw_vel, pitch_vel),:] = reading.data

    # Standard form: P = (I - K*H)*P
    # self.covar = (self.identity - kalman_gain*reading.obs_model) * self.covar

    # Use the Joseph form for numerical stability: P = (I-K*H)*P*(I - K*H)' + K*R*K'
    aux_mtrx = (self.identity - kalman_gain * reading.obs_model)
    self.covar = aux_mtrx * self.covar * aux_mtrx.T + kalman_gain * reading.covar * kalman_gain.T

    if self.DEBUG:
      print "After update"
      print "state\n", self.state
      print "covar:\n",self.covar

  def update_scalar(self, reading):
    # like update but knowing that measurment is a scalar
    # this avoids matrix inversions and speeds up (surprisingly) drived.py a lot

    # innovation = reading.data - np.matmul(reading.obs_model, self.state)   
    # innovation_covar = np.matmul(np.matmul(reading.obs_model, self.covar), reading.obs_model.T) + reading.covar    
    # kalman_gain = np.matmul(self.covar, reading.obs_model.T)/innovation_covar   
    # self.state += np.matmul(kalman_gain, innovation)   
    # aux_mtrx = self.identity - np.matmul(kalman_gain, reading.obs_model)    
    # self.covar =  np.matmul(aux_mtrx, np.matmul(self.covar, aux_mtrx.T)) + np.matmul(kalman_gain, np.matmul(reading.covar, kalman_gain.T))

    # written without np.matmul
    es = np.einsum
    ABC_T = "ij,jk,lk->il"
    AB_T = "ij,kj->ik"
    AB = "ij,jk->ik"
    innovation = reading.data - es(AB, reading.obs_model, self.state)
    innovation_covar = es(ABC_T, reading.obs_model, self.covar,
                          reading.obs_model) + reading.covar
    kalman_gain = es(AB_T, self.covar, reading.obs_model) / innovation_covar

    self.state += es(AB, kalman_gain, innovation)
    aux_mtrx = self.identity - es(AB, kalman_gain, reading.obs_model)
    self.covar = es(ABC_T, aux_mtrx, self.covar, aux_mtrx) + \
                 es(ABC_T, kalman_gain, reading.covar, kalman_gain)

  # Prediction update
  def predict(self, dt):
    es = np.einsum
    ABC_T = "ij,jk,lk->il"
    AB = "ij,jk->ik"

    # State update
    transfer_fun, transfer_fun_jacobian = self.calc_transfer_fun(dt)

    # self.state = np.matmul(transfer_fun, self.state)
    # self.covar = np.matmul(np.matmul(transfer_fun_jacobian, self.covar), transfer_fun_jacobian.T) + self.process_noise * dt

    # x = f(x, u), written in the form x = A(x, u)*x
    self.state = es(AB, transfer_fun, self.state)

    # P = J*P*J' + Q
    self.covar = es(ABC_T, transfer_fun_jacobian, self.covar,
                    transfer_fun_jacobian) + self.process_noise * dt  #!dt

    #! Clip covariance to avoid explosions
    self.covar = np.clip(self.covar,-1e10,1e10)
    
  @abc.abstractmethod
  def calc_transfer_fun(self, dt):
    """Return a tuple with the transfer function and transfer function jacobian
    The transfer function and jacobian should both be a numpy matrix of size DIMSxDIMS

    The transfer function matrix A should satisfy the state-update equation
      x_(k+1) = A * x_k

    The jacobian J is the direct jacobian A*x_k. For linear systems J=A.

    Current implementations calculate A and J as functions of state. Control input
      can be added trivially by adding a control parameter to predict() and calc_tranfer_update(),
      and using it during calcualtion of A and J
    """

class FastEKF1D(EKF):
  """Fast version of EKF for 1D problems with scalar readings."""

  def __init__(self, dt, var_init, Q):
    super(FastEKF1D, self).__init__(False)
    self.state = [0, 0]
    self.covar = [var_init, var_init, 0]

    # Process Noise
    self.dtQ0 = dt * Q[0]
    self.dtQ1 = dt * Q[1]

  def update(self, reading):
    raise NotImplementedError

  def update_scalar(self, reading):
    # TODO(mgraczyk): Delete this for speed.
    # assert np.all(reading.obs_model == [1, 0])

    rcov = reading.covar[0, 0]

    x = self.state
    S = self.covar

    innovation = reading.data - x[0]
    innovation_covar = S[0] + rcov

    k0 = S[0] / innovation_covar
    k1 = S[2] / innovation_covar

    x[0] += k0 * innovation
    x[1] += k1 * innovation

    mk = 1 - k0
    S[1] += k1 * (k1 * (S[0] + rcov) - 2 * S[2])
    S[2] = mk * (S[2] - k1 * S[0]) + rcov * k0 * k1
    S[0] = mk * mk * S[0] + rcov * k0 * k0

  def predict(self, dt):
    # State update
    x = self.state

    x[0] += dt * x[1]

    # P = J*P*J' + Q
    S = self.covar
    S[0] += dt * (2 * S[2] + dt * S[1]) + self.dtQ0
    S[2] += dt * S[1]
    S[1] += self.dtQ1

    # Clip covariance to avoid explosions
    S = max(-1e10, min(S, 1e10))

  def calc_transfer_fun(self, dt):
    tf = np.identity(2)
    tf[0, 1] = dt
    tfj = tf
    return tf, tfj
