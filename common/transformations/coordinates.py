import numpy as np
"""
Coordinate transformation module. All methods accept arrays as input
with each row as a position.
"""



a = 6378137
b = 6356752.3142
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001


def geodetic2ecef(geodetic):
  geodetic = np.array(geodetic)
  input_shape = geodetic.shape
  geodetic = np.atleast_2d(geodetic)
  lat = (np.pi/180)*geodetic[:,0]
  lon = (np.pi/180)*geodetic[:,1]
  alt = geodetic[:,2]

  xi = np.sqrt(1 - esq * np.sin(lat)**2)
  x = (a / xi + alt) * np.cos(lat) * np.cos(lon)
  y = (a / xi + alt) * np.cos(lat) * np.sin(lon)
  z = (a / xi * (1 - esq) + alt) * np.sin(lat)
  ecef = np.array([x, y, z]).T
  return ecef.reshape(input_shape)


def ecef2geodetic(ecef):
  """
  Convert ECEF coordinates to geodetic using ferrari's method
  """
  def ferrari(x, y, z):
    # ferrari's method
    r = np.sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * esq * esq * P)
    r_0 =  -(P * esq * r) / (1 + Q) + np.sqrt(0.5 * a * a*(1 + 1.0 / Q) - \
          P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - esq * r_0), 2) + z * z)
    V = np.sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = (180/np.pi)*np.arctan((z + e1sq * Z_0) / r)
    lon = (180/np.pi)*np.arctan2(y, x)
    return lat, lon, h

  geodetic = []
  ecef = np.array(ecef)
  input_shape = ecef.shape
  ecef = np.atleast_2d(ecef)
  for p in ecef:
    geodetic.append(ferrari(*p))
  geodetic = np.array(geodetic)
  return geodetic.reshape(input_shape)



class LocalCoord(object):
  """
   Allows conversions to local frames. In this case NED.
   That is: North East Down from the start position in
   meters.
  """
  def __init__(self, init_geodetic, init_ecef):
    self.init_ecef = init_ecef
    lat, lon, _ = (np.pi/180)*np.array(init_geodetic)
    self.ned2ecef_matrix = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
                                     [-np.sin(lat)*np.sin(lon), np.cos(lon), -np.cos(lat)*np.sin(lon)],
                                     [np.cos(lat), 0, -np.sin(lat)]])
    self.ecef2ned_matrix = self.ned2ecef_matrix.T

  @classmethod
  def from_geodetic(cls, init_geodetic):
    init_ecef = geodetic2ecef(init_geodetic)
    return LocalCoord(init_geodetic, init_ecef)

  @classmethod
  def from_ecef(cls, init_ecef):
    init_geodetic = ecef2geodetic(init_ecef)
    return LocalCoord(init_geodetic, init_ecef)


  def ecef2ned(self, ecef):
    ecef = np.array(ecef)
    return np.dot(self.ecef2ned_matrix, (ecef - self.init_ecef).T).T

  def ned2ecef(self, ned):
    ned = np.array(ned)
    # Transpose so that init_ecef will broadcast correctly for 1d or 2d ned.
    return (np.dot(self.ned2ecef_matrix, ned.T).T + self.init_ecef)

  def geodetic2ned(self, geodetic):
    ecef = geodetic2ecef(geodetic)
    return self.ecef2ned(ecef)

  def ned2geodetic(self, ned):
    ecef = self.ned2ecef(ned)
    return ecef2geodetic(ecef)
