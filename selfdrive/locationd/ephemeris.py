def GET_FIELD_U(w, nb, pos):
  return (((w) >> (pos)) & ((1 << (nb)) - 1))


def twos_complement(v, nb):
  sign = v >> (nb - 1)
  value = v
  if sign != 0:
    value = value - (1 << nb)
  return value


def GET_FIELD_S(w, nb, pos):
  v = GET_FIELD_U(w, nb, pos)
  return twos_complement(v, nb)


def extract_uint8(v, b):
  return (v >> (8 * (3 - b))) & 255

def extract_int8(v, b):
  value = extract_uint8(v, b)
  if value > 127:
    value -= 256
  return value

class EphemerisData:
  '''container for parsing a AID_EPH message
    Thanks to Sylvain Munaut <tnt@246tNt.com>
    http://cgit.osmocom.org/cgit/osmocom-lcs/tree/gps.c
on of this parser

    See IS-GPS-200F.pdf Table 20-III for the field meanings, scaling factors and
    field widths
    '''

  def __init__(self, svId, subframes):
    from math import pow
    self.svId = svId
    week_no = GET_FIELD_U(subframes[1][2+0], 10, 20)
    # code_on_l2 = GET_FIELD_U(subframes[1][0],  2, 12)
    # sv_ura     = GET_FIELD_U(subframes[1][0],  4,  8)
    # sv_health  = GET_FIELD_U(subframes[1][0],  6,  2)
    # l2_p_flag  = GET_FIELD_U(subframes[1][1],  1, 23)
    t_gd = GET_FIELD_S(subframes[1][2+4], 8, 6)
    iodc = (GET_FIELD_U(subframes[1][2+0], 2, 6) << 8) | GET_FIELD_U(
      subframes[1][2+5], 8, 22)

    t_oc = GET_FIELD_U(subframes[1][2+5], 16, 6)
    a_f2 = GET_FIELD_S(subframes[1][2+6], 8, 22)
    a_f1 = GET_FIELD_S(subframes[1][2+6], 16, 6)
    a_f0 = GET_FIELD_S(subframes[1][2+7], 22, 8)

    c_rs = GET_FIELD_S(subframes[2][2+0], 16, 6)
    delta_n = GET_FIELD_S(subframes[2][2+1], 16, 14)
    m_0 = (GET_FIELD_S(subframes[2][2+1], 8, 6) << 24) | GET_FIELD_U(
      subframes[2][2+2], 24, 6)
    c_uc = GET_FIELD_S(subframes[2][2+3], 16, 14)
    e = (GET_FIELD_U(subframes[2][2+3], 8, 6) << 24) | GET_FIELD_U(subframes[2][2+4], 24, 6)
    c_us = GET_FIELD_S(subframes[2][2+5], 16, 14)
    a_powhalf = (GET_FIELD_U(subframes[2][2+5], 8, 6) << 24) | GET_FIELD_U(
      subframes[2][2+6], 24, 6)
    t_oe = GET_FIELD_U(subframes[2][2+7], 16, 14)
    # fit_flag   = GET_FIELD_U(subframes[2][7],  1,  7)

    c_ic = GET_FIELD_S(subframes[3][2+0], 16, 14)
    omega_0 = (GET_FIELD_S(subframes[3][2+0], 8, 6) << 24) | GET_FIELD_U(
      subframes[3][2+1], 24, 6)
    c_is = GET_FIELD_S(subframes[3][2+2], 16, 14)
    i_0 = (GET_FIELD_S(subframes[3][2+2], 8, 6) << 24) | GET_FIELD_U(
      subframes[3][2+3], 24, 6)
    c_rc = GET_FIELD_S(subframes[3][2+4], 16, 14)
    w = (GET_FIELD_S(subframes[3][2+4], 8, 6) << 24) | GET_FIELD_U(subframes[3][5], 24, 6)
    omega_dot = GET_FIELD_S(subframes[3][2+6], 24, 6)
    idot = GET_FIELD_S(subframes[3][2+7], 14, 8)

    self._rsvd1 = GET_FIELD_U(subframes[1][2+1], 23, 6)
    self._rsvd2 = GET_FIELD_U(subframes[1][2+2], 24, 6)
    self._rsvd3 = GET_FIELD_U(subframes[1][2+3], 24, 6)
    self._rsvd4 = GET_FIELD_U(subframes[1][2+4], 16, 14)
    self.aodo = GET_FIELD_U(subframes[2][2+7], 5, 8)

    # Definition of Pi used in the GPS coordinate system
    gpsPi = 3.1415926535898

    # now form variables in radians, meters and seconds etc
    self.Tgd = t_gd * pow(2, -31)
    self.A = pow(a_powhalf * pow(2, -19), 2.0)
    self.cic = c_ic * pow(2, -29)
    self.cis = c_is * pow(2, -29)
    self.crc = c_rc * pow(2, -5)
    self.crs = c_rs * pow(2, -5)
    self.cuc = c_uc * pow(2, -29)
    self.cus = c_us * pow(2, -29)
    self.deltaN = delta_n * pow(2, -43) * gpsPi
    self.ecc = e * pow(2, -33)
    self.i0 = i_0 * pow(2, -31) * gpsPi
    self.idot = idot * pow(2, -43) * gpsPi
    self.M0 = m_0 * pow(2, -31) * gpsPi
    self.omega = w * pow(2, -31) * gpsPi
    self.omega_dot = omega_dot * pow(2, -43) * gpsPi
    self.omega0 = omega_0 * pow(2, -31) * gpsPi
    self.toe = t_oe * pow(2, 4)

    # clock correction information
    self.toc = t_oc * pow(2, 4)
    self.gpsWeek = week_no
    self.af0 = a_f0 * pow(2, -31)
    self.af1 = a_f1 * pow(2, -43)
    self.af2 = a_f2 * pow(2, -55)

    iode1 = GET_FIELD_U(subframes[2][2+0], 8, 22)
    iode2 = GET_FIELD_U(subframes[3][2+7], 8, 22)
    self.valid = (iode1 == iode2) and (iode1 == (iodc & 0xff))
    self.iode = iode1

    if GET_FIELD_U(subframes[4][2+0], 2, 28) != 1:
      raise RuntimeError("subframe 4 not of type 1")
    if GET_FIELD_U(subframes[5][2+0], 2, 28) != 1:
      raise RuntimeError("subframe 5 not of type 1")
    print 'page :', GET_FIELD_U(subframes[4][2+0], 6, 22)
    if GET_FIELD_U(subframes[4][2+0], 6, 22) == 56:
      a0 = GET_FIELD_S(subframes[4][2], 8, 14) * pow(2, -30)
      a1 = GET_FIELD_S(subframes[4][2], 8, 6) * pow(2, -27)
      a2 = GET_FIELD_S(subframes[4][3], 8, 22) * pow(2, -24)
      a3 = GET_FIELD_S(subframes[4][3], 8, 14) * pow(2, -24)
      b0 = GET_FIELD_S(subframes[4][3], 8, 6) * pow(2, 11)
      b1 = GET_FIELD_S(subframes[4][4], 8, 22) * pow(2, 14)
      b2 = GET_FIELD_S(subframes[4][4], 8, 14) * pow(2, 16)
      b3 = GET_FIELD_S(subframes[4][4], 8, 6) * pow(2, 16)

      self.ionoAlpha = [a0, a1, a2, a3]
      self.ionoBeta = [b0, b1, b2, b3]
      self.ionoCoeffsValid = True
    else:
      self.ionoAlpha = []
      self.ionoBeta = []
      self.ionoCoeffsValid = False


