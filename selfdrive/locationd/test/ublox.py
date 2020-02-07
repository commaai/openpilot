#!/usr/bin/env python3
'''
UBlox binary protocol handling

Copyright Andrew Tridgell, October 2012
Released under GNU GPL version 3 or later

WARNING: This code has originally intended for
ublox version 7, it has been adapted to work
for ublox version 8, not all functions may work.
'''


import struct
import time, os

# protocol constants
PREAMBLE1 = 0xb5
PREAMBLE2 = 0x62

# message classes
CLASS_NAV = 0x01
CLASS_RXM = 0x02
CLASS_INF = 0x04
CLASS_ACK = 0x05
CLASS_CFG = 0x06
CLASS_MON = 0x0A
CLASS_AID = 0x0B
CLASS_TIM = 0x0D
CLASS_ESF = 0x10

# ACK messages
MSG_ACK_NACK = 0x00
MSG_ACK_ACK = 0x01

# NAV messages
MSG_NAV_POSECEF = 0x1
MSG_NAV_POSLLH = 0x2
MSG_NAV_STATUS = 0x3
MSG_NAV_DOP = 0x4
MSG_NAV_SOL = 0x6
MSG_NAV_PVT = 0x7
MSG_NAV_POSUTM = 0x8
MSG_NAV_VELNED = 0x12
MSG_NAV_VELECEF = 0x11
MSG_NAV_TIMEGPS = 0x20
MSG_NAV_TIMEUTC = 0x21
MSG_NAV_CLOCK = 0x22
MSG_NAV_SVINFO = 0x30
MSG_NAV_AOPSTATUS = 0x60
MSG_NAV_DGPS = 0x31
MSG_NAV_DOP = 0x04
MSG_NAV_EKFSTATUS = 0x40
MSG_NAV_SBAS = 0x32
MSG_NAV_SOL = 0x06

# RXM messages
MSG_RXM_RAW = 0x15
MSG_RXM_SFRB = 0x11
MSG_RXM_SFRBX = 0x13
MSG_RXM_SVSI = 0x20
MSG_RXM_EPH = 0x31
MSG_RXM_ALM = 0x30
MSG_RXM_PMREQ = 0x41

# AID messages
MSG_AID_ALM = 0x30
MSG_AID_EPH = 0x31
MSG_AID_ALPSRV = 0x32
MSG_AID_AOP = 0x33
MSG_AID_DATA = 0x10
MSG_AID_ALP = 0x50
MSG_AID_DATA = 0x10
MSG_AID_HUI = 0x02
MSG_AID_INI = 0x01
MSG_AID_REQ = 0x00

# CFG messages
MSG_CFG_PRT = 0x00
MSG_CFG_ANT = 0x13
MSG_CFG_DAT = 0x06
MSG_CFG_EKF = 0x12
MSG_CFG_ESFGWT = 0x29
MSG_CFG_CFG = 0x09
MSG_CFG_USB = 0x1b
MSG_CFG_RATE = 0x08
MSG_CFG_SET_RATE = 0x01
MSG_CFG_NAV5 = 0x24
MSG_CFG_FXN = 0x0E
MSG_CFG_INF = 0x02
MSG_CFG_ITFM = 0x39
MSG_CFG_MSG = 0x01
MSG_CFG_NAVX5 = 0x23
MSG_CFG_NMEA = 0x17
MSG_CFG_NVS = 0x22
MSG_CFG_PM2 = 0x3B
MSG_CFG_PM = 0x32
MSG_CFG_RINV = 0x34
MSG_CFG_RST = 0x04
MSG_CFG_RXM = 0x11
MSG_CFG_SBAS = 0x16
MSG_CFG_TMODE2 = 0x3D
MSG_CFG_TMODE = 0x1D
MSG_CFG_TPS = 0x31
MSG_CFG_TP = 0x07
MSG_CFG_GNSS = 0x3E
MSG_CFG_ODO = 0x1E

# ESF messages
MSG_ESF_MEAS = 0x02
MSG_ESF_STATUS = 0x10

# INF messages
MSG_INF_DEBUG = 0x04
MSG_INF_ERROR = 0x00
MSG_INF_NOTICE = 0x02
MSG_INF_TEST = 0x03
MSG_INF_WARNING = 0x01

# MON messages
MSG_MON_SCHD = 0x01
MSG_MON_HW = 0x09
MSG_MON_HW2 = 0x0B
MSG_MON_IO = 0x02
MSG_MON_MSGPP = 0x06
MSG_MON_RXBUF = 0x07
MSG_MON_RXR = 0x21
MSG_MON_TXBUF = 0x08
MSG_MON_VER = 0x04

# TIM messages
MSG_TIM_TP = 0x01
MSG_TIM_TM2 = 0x03
MSG_TIM_SVIN = 0x04
MSG_TIM_VRFY = 0x06

# port IDs
PORT_DDC = 0
PORT_SERIAL1 = 1
PORT_SERIAL2 = 2
PORT_USB = 3
PORT_SPI = 4

# dynamic models
DYNAMIC_MODEL_PORTABLE = 0
DYNAMIC_MODEL_STATIONARY = 2
DYNAMIC_MODEL_PEDESTRIAN = 3
DYNAMIC_MODEL_AUTOMOTIVE = 4
DYNAMIC_MODEL_SEA = 5
DYNAMIC_MODEL_AIRBORNE1G = 6
DYNAMIC_MODEL_AIRBORNE2G = 7
DYNAMIC_MODEL_AIRBORNE4G = 8

#reset items
RESET_HOT = 0
RESET_WARM = 1
RESET_COLD = 0xFFFF

RESET_HW = 0
RESET_SW = 1
RESET_SW_GPS = 2
RESET_HW_GRACEFUL = 4
RESET_GPS_STOP = 8
RESET_GPS_START = 9


class UBloxError(Exception):
  '''Ublox error class'''

  def __init__(self, msg):
    Exception.__init__(self, msg)
    self.message = msg


class UBloxAttrDict(dict):
  '''allow dictionary members as attributes'''

  def __init__(self):
    dict.__init__(self)

  def __getattr__(self, name):
    try:
      return self.__getitem__(name)
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    if name in self.__dict__:
      # allow set on normal attributes
      dict.__setattr__(self, name, value)
    else:
      self.__setitem__(name, value)


def ArrayParse(field):
  '''parse an array descriptor'''
  arridx = field.find('[')
  if arridx == -1:
    return (field, -1)
  alen = int(field[arridx + 1:-1])
  fieldname = field[:arridx]
  return (fieldname, alen)


class UBloxDescriptor:
  '''class used to describe the layout of a UBlox message'''

  def __init__(self,
               name,
               msg_format,
               fields=None,
               count_field=None,
               format2=None,
               fields2=None):
    if fields is None:
      fields = []

    self.name = name
    self.msg_format = msg_format
    self.fields = fields
    self.count_field = count_field
    self.format2 = format2
    self.fields2 = fields2

  def unpack(self, msg):
    '''unpack a UBloxMessage, creating the .fields and ._recs attributes in msg'''
    msg._fields = {}

    # unpack main message blocks. A comm
    formats = self.msg_format.split(',')
    buf = msg._buf[6:-2]
    count = 0
    msg._recs = []
    fields = self.fields[:]

    for fmt in formats:
      size1 = struct.calcsize(fmt)
      if size1 > len(buf):
        raise UBloxError("%s INVALID_SIZE1=%u" % (self.name, len(buf)))
      f1 = list(struct.unpack(fmt, buf[:size1]))
      i = 0
      while i < len(f1):
        field = fields.pop(0)
        (fieldname, alen) = ArrayParse(field)
        if alen == -1:
          msg._fields[fieldname] = f1[i]
          if self.count_field == fieldname:
            count = int(f1[i])
          i += 1
        else:
          msg._fields[fieldname] = [0] * alen
          for a in range(alen):
            msg._fields[fieldname][a] = f1[i]
            i += 1
      buf = buf[size1:]
      if len(buf) == 0:
        break

    if self.count_field == '_remaining':
      count = len(buf) // struct.calcsize(self.format2)

    if count == 0:
      msg._unpacked = True
      if len(buf) != 0:
        raise UBloxError("EXTRA_BYTES=%u" % len(buf))
      return

    size2 = struct.calcsize(self.format2)
    for c in range(count):
      r = UBloxAttrDict()
      if size2 > len(buf):
        raise UBloxError("INVALID_SIZE=%u, " % len(buf))
      f2 = list(struct.unpack(self.format2, buf[:size2]))
      for i in range(len(self.fields2)):
        r[self.fields2[i]] = f2[i]
      buf = buf[size2:]
      msg._recs.append(r)
    if len(buf) != 0:
      raise UBloxError("EXTRA_BYTES=%u" % len(buf))
    msg._unpacked = True

  def pack(self, msg, msg_class=None, msg_id=None):
    '''pack a UBloxMessage from the .fields and ._recs attributes in msg'''
    f1 = []
    if msg_class is None:
      msg_class = msg.msg_class()
    if msg_id is None:
      msg_id = msg.msg_id()
    msg._buf = ''

    fields = self.fields[:]
    for f in fields:
      (fieldname, alen) = ArrayParse(f)
      if not fieldname in msg._fields:
        break
      if alen == -1:
        f1.append(msg._fields[fieldname])
      else:
        for a in range(alen):
          f1.append(msg._fields[fieldname][a])
    try:
      # try full length message
      fmt = self.msg_format.replace(',', '')
      msg._buf = struct.pack(fmt, *tuple(f1))
    except Exception:
      # try without optional part
      fmt = self.msg_format.split(',')[0]
      msg._buf = struct.pack(fmt, *tuple(f1))

    length = len(msg._buf)
    if msg._recs:
      length += len(msg._recs) * struct.calcsize(self.format2)
    header = struct.pack('<BBBBH', PREAMBLE1, PREAMBLE2, msg_class, msg_id, length)
    msg._buf = header + msg._buf

    for r in msg._recs:
      f2 = []
      for f in self.fields2:
        f2.append(r[f])
      msg._buf += struct.pack(self.format2, *tuple(f2))
    msg._buf += struct.pack('<BB', *msg.checksum(data=msg._buf[2:]))

  def format(self, msg):
    '''return a formatted string for a message'''
    if not msg._unpacked:
      self.unpack(msg)
    ret = self.name + ': '
    for f in self.fields:
      (fieldname, alen) = ArrayParse(f)
      if not fieldname in msg._fields:
        continue
      v = msg._fields[fieldname]
      if isinstance(v, list):
        ret += '%s=[' % fieldname
        for a in range(alen):
          ret += '%s, ' % v[a]
        ret = ret[:-2] + '], '
      elif isinstance(v, str):
        ret += '%s="%s", ' % (f, v.rstrip(' \0'))
      else:
        ret += '%s=%s, ' % (f, v)
    for r in msg._recs:
      ret += '[ '
      for f in self.fields2:
        v = r[f]
        ret += '%s=%s, ' % (f, v)
      ret = ret[:-2] + ' ], '
    return ret[:-2]


# list of supported message types.
msg_types = {
  (CLASS_ACK, MSG_ACK_ACK):
  UBloxDescriptor('ACK_ACK', '<BB', ['clsID', 'msgID']),
  (CLASS_ACK, MSG_ACK_NACK):
  UBloxDescriptor('ACK_NACK', '<BB', ['clsID', 'msgID']),
  (CLASS_CFG, MSG_CFG_USB):
  UBloxDescriptor('CFG_USB', '<HHHHHH32s32s32s', [
    'vendorID', 'productID', 'reserved1', 'reserved2', 'powerConsumption', 'flags',
    'vendorString', 'productString', 'serialNumber'
  ]),
  (CLASS_CFG, MSG_CFG_PRT):
  UBloxDescriptor('CFG_PRT', '<BBHIIHHHH', [
    'portID', 'reserved0', 'txReady', 'mode', 'baudRate', 'inProtoMask', 'outProtoMask',
    'reserved4', 'reserved5'
  ]),
  (CLASS_CFG, MSG_CFG_CFG):
  UBloxDescriptor('CFG_CFG', '<III,B',
                  ['clearMask', 'saveMask', 'loadMask', 'deviceMask']),
  (CLASS_CFG, MSG_CFG_RXM):
  UBloxDescriptor('CFG_RXM', '<BB',
                  ['reserved1', 'lpMode']),
  (CLASS_CFG, MSG_CFG_RST):
  UBloxDescriptor('CFG_RST', '<HBB', ['navBbrMask ', 'resetMode', 'reserved1']),
  (CLASS_CFG, MSG_CFG_SBAS):
  UBloxDescriptor('CFG_SBAS', '<BBBBI',
                  ['mode', 'usage', 'maxSBAS', 'scanmode2', 'scanmode1']),
  (CLASS_CFG, MSG_CFG_GNSS):
  UBloxDescriptor('CFG_GNSS', '<BBBB',
                  ['msgVer', 'numTrkChHw', 'numTrkChUse',
                   'numConfigBlocks'], 'numConfigBlocks', '<BBBBI',
                  ['gnssId', 'resTrkCh', 'maxTrkCh', 'reserved1', 'flags']),
  (CLASS_CFG, MSG_CFG_RATE):
  UBloxDescriptor('CFG_RATE', '<HHH', ['measRate', 'navRate', 'timeRef']),
  (CLASS_CFG, MSG_CFG_MSG):
  UBloxDescriptor('CFG_MSG', '<BB6B', ['msgClass', 'msgId', 'rates[6]']),
  (CLASS_NAV, MSG_NAV_POSLLH):
  UBloxDescriptor('NAV_POSLLH', '<IiiiiII',
                  ['iTOW', 'Longitude', 'Latitude', 'height', 'hMSL', 'hAcc', 'vAcc']),
  (CLASS_NAV, MSG_NAV_VELNED):
  UBloxDescriptor('NAV_VELNED', '<IiiiIIiII', [
    'iTOW', 'velN', 'velE', 'velD', 'speed', 'gSpeed', 'heading', 'sAcc', 'cAcc'
  ]),
  (CLASS_NAV, MSG_NAV_DOP):
  UBloxDescriptor('NAV_DOP', '<IHHHHHHH',
                  ['iTOW', 'gDOP', 'pDOP', 'tDOP', 'vDOP', 'hDOP', 'nDOP', 'eDOP']),
  (CLASS_NAV, MSG_NAV_STATUS):
  UBloxDescriptor('NAV_STATUS', '<IBBBBII',
                  ['iTOW', 'gpsFix', 'flags', 'fixStat', 'flags2', 'ttff', 'msss']),
  (CLASS_NAV, MSG_NAV_SOL):
  UBloxDescriptor('NAV_SOL', '<IihBBiiiIiiiIHBBI', [
    'iTOW', 'fTOW', 'week', 'gpsFix', 'flags', 'ecefX', 'ecefY', 'ecefZ', 'pAcc',
    'ecefVX', 'ecefVY', 'ecefVZ', 'sAcc', 'pDOP', 'reserved1', 'numSV', 'reserved2'
  ]),
  (CLASS_NAV, MSG_NAV_PVT):
  UBloxDescriptor('NAV_PVT', '<IHBBBBBBIiBBBBiiiiIIiiiiiIIH6BihH', [
    'iTOW', 'year', 'month', 'day', 'hour', 'min', 'sec', 'valid', 'tAcc', 'nano',
    'fixType', 'flags', 'flags2', 'numSV', 'lon', 'lat', 'height', 'hMSL', 'hAcc', 'vAcc',
    'velN', 'velE', 'velD', 'gSpeed', 'headMot', 'sAcc', 'headAcc', 'pDOP',
    'reserverd1[6]', 'headVeh', 'magDec', 'magAcc'
  ]),
  (CLASS_NAV, MSG_NAV_POSUTM):
  UBloxDescriptor('NAV_POSUTM', '<Iiiibb',
                  ['iTOW', 'East', 'North', 'Alt', 'Zone', 'Hem']),
  (CLASS_NAV, MSG_NAV_SBAS):
  UBloxDescriptor('NAV_SBAS', '<IBBbBBBBB', [
    'iTOW', 'geo', 'mode', 'sys', 'service', 'cnt', 'reserved01', 'reserved02',
    'reserved03'
  ], 'cnt', 'BBBBBBhHh', [
    'svid', 'flags', 'udre', 'svSys', 'svService', 'reserved1', 'prc', 'reserved2', 'ic'
  ]),
  (CLASS_NAV, MSG_NAV_POSECEF):
  UBloxDescriptor('NAV_POSECEF', '<IiiiI', ['iTOW', 'ecefX', 'ecefY', 'ecefZ', 'pAcc']),
  (CLASS_NAV, MSG_NAV_VELECEF):
  UBloxDescriptor('NAV_VELECEF', '<IiiiI', ['iTOW', 'ecefVX', 'ecefVY', 'ecefVZ',
                                            'sAcc']),
  (CLASS_NAV, MSG_NAV_TIMEGPS):
  UBloxDescriptor('NAV_TIMEGPS', '<IihbBI',
                  ['iTOW', 'fTOW', 'week', 'leapS', 'valid', 'tAcc']),
  (CLASS_NAV, MSG_NAV_TIMEUTC):
  UBloxDescriptor('NAV_TIMEUTC', '<IIiHBBBBBB', [
    'iTOW', 'tAcc', 'nano', 'year', 'month', 'day', 'hour', 'min', 'sec', 'valid'
  ]),
  (CLASS_NAV, MSG_NAV_CLOCK):
  UBloxDescriptor('NAV_CLOCK', '<IiiII', ['iTOW', 'clkB', 'clkD', 'tAcc', 'fAcc']),
  (CLASS_NAV, MSG_NAV_DGPS):
  UBloxDescriptor('NAV_DGPS', '<IihhBBH',
                  ['iTOW', 'age', 'baseId', 'baseHealth', 'numCh', 'status', 'reserved1'],
                  'numCh', '<BBHff', ['svid', 'flags', 'ageC', 'prc', 'prrc']),
  (CLASS_NAV, MSG_NAV_SVINFO):
  UBloxDescriptor('NAV_SVINFO', '<IBBH', ['iTOW', 'numCh', 'globalFlags',
                                          'reserved2'], 'numCh', '<BBBBBbhi',
                  ['chn', 'svid', 'flags', 'quality', 'cno', 'elev', 'azim', 'prRes']),
  (CLASS_RXM, MSG_RXM_SVSI):
  UBloxDescriptor('RXM_SVSI', '<IhBB', ['iTOW', 'week', 'numVis', 'numSV'], 'numSV',
                  '<BBhbB', ['svid', 'svFlag', 'azim', 'elev', 'age']),
  (CLASS_RXM, MSG_RXM_EPH):
  UBloxDescriptor('RXM_EPH', '<II , 8I 8I 8I',
                  ['svid', 'how', 'sf1d[8]', 'sf2d[8]', 'sf3d[8]']),
  (CLASS_AID, MSG_AID_EPH):
  UBloxDescriptor('AID_EPH', '<II , 8I 8I 8I',
                  ['svid', 'how', 'sf1d[8]', 'sf2d[8]', 'sf3d[8]']),
  (CLASS_AID, MSG_AID_HUI):
  UBloxDescriptor('AID_HUI', '<Iddi 6h 8f I',
                  ['health', 'utcA0', 'utcA1', 'utcTOW', 'utcWNT', 'utcLS', 'utcWNF',
                   'utcDN', 'utcLSF', 'utcSpare', 'klobA0', 'klobA1', 'klobA2', 'klobA3',
                   'klobB0', 'klobB1', 'klobB2', 'klobB3', 'flags']),
  (CLASS_AID, MSG_AID_AOP):
  UBloxDescriptor('AID_AOP', '<B47B , 48B 48B 48B',
                  ['svid', 'data[47]', 'optional0[48]', 'optional1[48]',
                   'optional1[48]']),
  (CLASS_RXM, MSG_RXM_RAW):
  UBloxDescriptor('RXM_RAW', '<dHbBB3B', [
    'rcvTow', 'week', 'leapS', 'numMeas', 'recStat', 'reserved1[3]'
  ], 'numMeas', '<ddfBBBBHBBBBBB', [
    'prMes', 'cpMes', 'doMes', 'gnssId', 'svId', 'sigId', 'freqId', 'locktime', 'cno',
    'prStdev', 'cpStdev', 'doStdev', 'trkStat', 'reserved3'
  ]),
  (CLASS_RXM, MSG_RXM_SFRB):
  UBloxDescriptor('RXM_SFRB', '<BB10I', ['chn', 'svid', 'dwrd[10]']),
  (CLASS_RXM, MSG_RXM_SFRBX):
  UBloxDescriptor('RXM_SFRBX', '<8B', ['gnssId', 'svid', 'reserved1', 'freqId', 'numWords',
      'reserved2', 'version', 'reserved3'], 'numWords', 'I', ['dwrd']),
  (CLASS_AID, MSG_AID_ALM):
  UBloxDescriptor('AID_ALM', '<II', '_remaining', 'I', ['dwrd']),
  (CLASS_RXM, MSG_RXM_ALM):
  UBloxDescriptor('RXM_ALM', '<II , 8I', ['svid', 'week', 'dwrd[8]']),
  (CLASS_CFG, MSG_CFG_ANT):
  UBloxDescriptor('CFG_ANT', '<HH', ['flags', 'pins']),
  (CLASS_CFG, MSG_CFG_ODO):
  UBloxDescriptor('CFG_ODO', '<B3BBB6BBB2BBB2B', [
    'version', 'reserved1[3]', 'flags', 'odoCfg', 'reserverd2[6]', 'cogMaxSpeed',
    'cogMaxPosAcc', 'reserved3[2]', 'velLpGain', 'cogLpGain', 'reserved[2]'
  ]),
  (CLASS_CFG, MSG_CFG_NAV5):
  UBloxDescriptor('CFG_NAV5', '<HBBiIbBHHHHBBIII', [
    'mask', 'dynModel', 'fixMode', 'fixedAlt', 'fixedAltVar', 'minElev', 'drLimit',
    'pDop', 'tDop', 'pAcc', 'tAcc', 'staticHoldThresh', 'dgpsTimeOut', 'reserved2',
    'reserved3', 'reserved4'
  ]),
  (CLASS_CFG, MSG_CFG_NAVX5):
  UBloxDescriptor('CFG_NAVX5', '<HHIBBBBBBBBBBHIBBBBBBHII', [
    'version', 'mask1', 'reserved0', 'reserved1', 'reserved2', 'minSVs', 'maxSVs',
    'minCNO', 'reserved5', 'iniFix3D', 'reserved6', 'reserved7', 'reserved8',
    'wknRollover', 'reserved9', 'reserved10', 'reserved11', 'usePPP', 'useAOP',
    'reserved12', 'reserved13', 'aopOrbMaxErr', 'reserved3', 'reserved4'
  ]),
  (CLASS_MON, MSG_MON_HW):
  UBloxDescriptor('MON_HW', '<IIIIHHBBBBIB17BHIII', [
    'pinSel', 'pinBank', 'pinDir', 'pinVal', 'noisePerMS', 'agcCnt', 'aStatus', 'aPower',
    'flags', 'reserved1', 'usedMask', 'VP[17]', 'jamInd', 'reserved3', 'pinInq', 'pullH',
    'pullL'
  ]),
  (CLASS_MON, MSG_MON_HW2):
  UBloxDescriptor('MON_HW2', '<bBbBB3BI8BI4B', [
    'ofsI', 'magI', 'ofsQ', 'magQ', 'cfgSource', 'reserved1[3]', 'lowLevCfg',
    'reserved2[8]', 'postStatus', 'reserved3[4]'
  ]),
  (CLASS_MON, MSG_MON_SCHD):
  UBloxDescriptor('MON_SCHD', '<IIIIHHHBB', [
    'tskRun', 'tskSchd', 'tskOvrr', 'tskReg', 'stack', 'stackSize', 'CPUIdle', 'flySly',
    'ptlSly'
  ]),
  (CLASS_MON, MSG_MON_VER):
  UBloxDescriptor('MON_VER', '<30s10s,30s', ['swVersion', 'hwVersion', 'romVersion'],
                  '_remaining', '30s', ['extension']),
  (CLASS_TIM, MSG_TIM_TP):
  UBloxDescriptor('TIM_TP', '<IIiHBB',
                  ['towMS', 'towSubMS', 'qErr', 'week', 'flags', 'reserved1']),
  (CLASS_TIM, MSG_TIM_TM2):
  UBloxDescriptor('TIM_TM2', '<BBHHHIIIII', [
    'ch', 'flags', 'count', 'wnR', 'wnF', 'towMsR', 'towSubMsR', 'towMsF', 'towSubMsF',
    'accEst'
  ]),
  (CLASS_TIM, MSG_TIM_SVIN):
  UBloxDescriptor('TIM_SVIN', '<IiiiIIBBH', [
    'dur', 'meanX', 'meanY', 'meanZ', 'meanV', 'obs', 'valid', 'active', 'reserved1'
  ]),
  (CLASS_INF, MSG_INF_ERROR):
  UBloxDescriptor('INF_ERR', '<18s', ['str']),
  (CLASS_INF, MSG_INF_DEBUG):
  UBloxDescriptor('INF_DEBUG', '<18s', ['str'])
}


class UBloxMessage:
  '''UBlox message class - holds a UBX binary message'''

  def __init__(self):
    self._buf = b""
    self._fields = {}
    self._recs = []
    self._unpacked = False
    self.debug_level = 1

  def __str__(self):
    '''format a message as a string'''
    if not self.valid():
      return 'UBloxMessage(INVALID)'
    type = self.msg_type()
    if type in msg_types:
      return msg_types[type].format(self)
    return 'UBloxMessage(UNKNOWN %s, %u)' % (str(type), self.msg_length())

  def as_dict(self):
    '''format a message as a string'''
    if not self.valid():
      return 'UBloxMessage(INVALID)'
    type = self.msg_type()
    if type in msg_types:
      return msg_types[type].format(self)
    return 'UBloxMessage(UNKNOWN %s, %u)' % (str(type), self.msg_length())

  def __getattr__(self, name):
    '''allow access to message fields'''
    try:
      return self._fields[name]
    except KeyError:
      if name == 'recs':
        return self._recs
      raise AttributeError(name)

  def __setattr__(self, name, value):
    '''allow access to message fields'''
    if name.startswith('_'):
      self.__dict__[name] = value
    else:
      self._fields[name] = value

  def have_field(self, name):
    '''return True if a message contains the given field'''
    return name in self._fields

  def debug(self, level, msg):
    '''write a debug message'''
    if self.debug_level >= level:
      print(msg)

  def unpack(self):
    '''unpack a message'''
    if not self.valid():
      raise UBloxError('INVALID MESSAGE')
    type = self.msg_type()
    if not type in msg_types:
      raise UBloxError('Unknown message %s length=%u' % (str(type), len(self._buf)))
    msg_types[type].unpack(self)
    return self._fields, self._recs

  def pack(self):
    '''pack a message'''
    if not self.valid():
      raise UBloxError('INVALID MESSAGE')
    type = self.msg_type()
    if not type in msg_types:
      raise UBloxError('Unknown message %s' % str(type))
    msg_types[type].pack(self)

  def name(self):
    '''return the short string name for a message'''
    if not self.valid():
      raise UBloxError('INVALID MESSAGE')
    type = self.msg_type()
    if not type in msg_types:
      raise UBloxError('Unknown message %s length=%u' % (str(type), len(self._buf)))
    return msg_types[type].name

  def msg_class(self):
    '''return the message class'''
    return self._buf[2]

  def msg_id(self):
    '''return the message id within the class'''
    return self._buf[3]

  def msg_type(self):
    '''return the message type tuple (class, id)'''
    return (self.msg_class(), self.msg_id())

  def msg_length(self):
    '''return the payload length'''
    (payload_length, ) = struct.unpack('<H', self._buf[4:6])
    return payload_length

  def valid_so_far(self):
    '''check if the message is valid so far'''
    if len(self._buf) > 0 and self._buf[0] != PREAMBLE1:
      return False
    if len(self._buf) > 1 and self._buf[1] != PREAMBLE2:
      self.debug(1, "bad pre2")
      return False
    if self.needed_bytes() == 0 and not self.valid():
      if len(self._buf) > 8:
        self.debug(1, "bad checksum len=%u needed=%u" % (len(self._buf),
                                                         self.needed_bytes()))
      else:
        self.debug(1, "bad len len=%u needed=%u" % (len(self._buf), self.needed_bytes()))
      return False
    return True

  def add(self, bytes):
    '''add some bytes to a message'''
    self._buf += bytes
    while not self.valid_so_far() and len(self._buf) > 0:
      '''handle corrupted streams'''
      self._buf = self._buf[1:]
    if self.needed_bytes() < 0:
      self._buf = ""

  def checksum(self, data=None):
    '''return a checksum tuple for a message'''
    if data is None:
      data = self._buf[2:-2]
    #cs = 0
    ck_a = 0
    ck_b = 0
    for i in data:
      ck_a = (ck_a + i) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    return (ck_a, ck_b)

  def valid_checksum(self):
    '''check if the checksum is OK'''
    (ck_a, ck_b) = self.checksum()
    #d = self._buf[2:-2]
    (ck_a2, ck_b2) = struct.unpack('<BB', self._buf[-2:])
    return ck_a == ck_a2 and ck_b == ck_b2

  def needed_bytes(self):
    '''return number of bytes still needed'''
    if len(self._buf) < 6:
      return 8 - len(self._buf)
    return self.msg_length() + 8 - len(self._buf)

  def valid(self):
    '''check if a message is valid'''
    return len(self._buf) >= 8 and self.needed_bytes() == 0 and self.valid_checksum()


class UBlox:
  '''main UBlox control class.

    port can be a file (for reading only) or a serial device
    '''

  def __init__(self, port, baudrate=115200, timeout=0, panda=False, grey=False):

    self.serial_device = port
    self.baudrate = baudrate
    self.use_sendrecv = False
    self.read_only = False
    self.debug_level = 0

    if panda:
      from panda import Panda, PandaSerial

      self.panda = Panda()

      # resetting U-Blox module
      self.panda.set_esp_power(0)
      time.sleep(0.1)
      self.panda.set_esp_power(1)
      time.sleep(0.5)

      # can't set above 9600 now...
      self.baudrate = 9600
      self.dev = PandaSerial(self.panda, 1, self.baudrate)

      self.baudrate = 460800
      print("upping baud:",self.baudrate)
      self.send_nmea("$PUBX,41,1,0007,0003,%u,0" % self.baudrate)
      time.sleep(0.1)

      self.dev = PandaSerial(self.panda, 1, self.baudrate)
    elif grey:
      import cereal.messaging as messaging

      class BoarddSerial():
        def __init__(self):
          self.ubloxRaw = messaging.sub_sock('ubloxRaw')
          self.buf = ""

        def read(self, n):
          for msg in messaging.drain_sock(self.ubloxRaw, len(self.buf) < n):
            self.buf += msg.ubloxRaw
          ret = self.buf[:n]
          self.buf = self.buf[n:]
          return ret


        def write(self, dat):
          pass

      self.dev = BoarddSerial()
    else:
      if self.serial_device.startswith("tcp:"):
        import socket
        a = self.serial_device.split(':')
        destination_addr = (a[1], int(a[2]))
        self.dev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dev.connect(destination_addr)
        self.dev.setblocking(1)
        self.dev.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self.use_sendrecv = True
      elif os.path.isfile(self.serial_device):
        self.read_only = True
        self.dev = open(self.serial_device, mode='rb')
      else:
        import serial
        self.dev = serial.Serial(
          self.serial_device,
          baudrate=self.baudrate,
          dsrdtr=False,
          rtscts=False,
          xonxoff=False,
          timeout=timeout)

    self.logfile = None
    self.log = None
    self.preferred_dynamic_model = None
    self.preferred_usePPP = None
    self.preferred_dgps_timeout = None

  def close(self):
    '''close the device'''
    self.dev.close()
    self.dev = None

  def set_debug(self, debug_level):
    '''set debug level'''
    self.debug_level = debug_level

  def debug(self, level, msg):
    '''write a debug message'''
    if self.debug_level >= level:
      print(msg)

  def set_logfile(self, logfile, append=False):
    '''setup logging to a file'''
    if self.log is not None:
      self.log.close()
      self.log = None
    self.logfile = logfile
    if self.logfile is not None:
      if append:
        mode = 'ab'
      else:
        mode = 'wb'
      self.log = open(self.logfile, mode=mode)

  def set_preferred_dynamic_model(self, model):
    '''set the preferred dynamic model for receiver'''
    self.preferred_dynamic_model = model
    if model is not None:
      self.configure_poll(CLASS_CFG, MSG_CFG_NAV5)

  def set_preferred_dgps_timeout(self, timeout):
    '''set the preferred DGPS timeout for receiver'''
    self.preferred_dgps_timeout = timeout
    if timeout is not None:
      self.configure_poll(CLASS_CFG, MSG_CFG_NAV5)

  def set_preferred_usePPP(self, usePPP):
    '''set the preferred usePPP setting for the receiver'''
    if usePPP is None:
      self.preferred_usePPP = None
      return
    self.preferred_usePPP = int(usePPP)
    self.configure_poll(CLASS_CFG, MSG_CFG_NAVX5)

  def nmea_checksum(self, msg):
    d = msg[1:]
    cs = 0
    for i in d:
      cs ^= ord(i)
    return cs

  def write(self, buf):
    '''write some bytes'''
    if not self.read_only:
      if self.use_sendrecv:
        return self.dev.send(buf)
      if type(buf) == str:
        return self.dev.write(str.encode(buf))
      else:
        return self.dev.write(buf)

  def read(self, n):
    '''read some bytes'''
    if self.use_sendrecv:
      import socket
      try:
        return self.dev.recv(n)
      except socket.error:
        return ''
    return self.dev.read(n)

  def send_nmea(self, msg):
    if not self.read_only:
      s = msg + "*%02X" % self.nmea_checksum(msg) + "\r\n"
      self.write(s)

  def set_binary(self):
    '''put a UBlox into binary mode using a NMEA string'''
    if not self.read_only:
      print("try set binary at %u" % self.baudrate)
      self.send_nmea("$PUBX,41,0,0007,0001,%u,0" % self.baudrate)
      self.send_nmea("$PUBX,41,1,0007,0001,%u,0" % self.baudrate)
      self.send_nmea("$PUBX,41,2,0007,0001,%u,0" % self.baudrate)
      self.send_nmea("$PUBX,41,3,0007,0001,%u,0" % self.baudrate)
      self.send_nmea("$PUBX,41,4,0007,0001,%u,0" % self.baudrate)
      self.send_nmea("$PUBX,41,5,0007,0001,%u,0" % self.baudrate)

  def disable_nmea(self):
    ''' stop sending all types of nmea messages '''
    self.send_nmea("$PUBX,40,GSV,1,1,1,1,1,0")
    self.send_nmea("$PUBX,40,GGA,0,0,0,0,0,0")
    self.send_nmea("$PUBX,40,GSA,0,0,0,0,0,0")
    self.send_nmea("$PUBX,40,VTG,0,0,0,0,0,0")
    self.send_nmea("$PUBX,40,TXT,0,0,0,0,0,0")
    self.send_nmea("$PUBX,40,RMC,0,0,0,0,0,0")

  def seek_percent(self, pct):
    '''seek to the given percentage of a file'''
    self.dev.seek(0, 2)
    filesize = self.dev.tell()
    self.dev.seek(pct * 0.01 * filesize)

  def special_handling(self, msg):
    '''handle automatic configuration changes'''
    if msg.name() == 'CFG_NAV5':
      msg.unpack()
      sendit = False
      pollit = False
      if self.preferred_dynamic_model is not None and msg.dynModel != self.preferred_dynamic_model:
        msg.dynModel = self.preferred_dynamic_model
        sendit = True
        pollit = True
      if self.preferred_dgps_timeout is not None and msg.dgpsTimeOut != self.preferred_dgps_timeout:
        msg.dgpsTimeOut = self.preferred_dgps_timeout
        self.debug(2, "Setting dgpsTimeOut=%u" % msg.dgpsTimeOut)
        sendit = True
        # we don't re-poll for this one, as some receivers refuse to set it
      if sendit:
        msg.pack()
        self.send(msg)
        if pollit:
          self.configure_poll(CLASS_CFG, MSG_CFG_NAV5)
    if msg.name() == 'CFG_NAVX5' and self.preferred_usePPP is not None:
      msg.unpack()
      if msg.usePPP != self.preferred_usePPP:
        msg.usePPP = self.preferred_usePPP
        msg.mask = 1 << 13
        msg.pack()
        self.send(msg)
        self.configure_poll(CLASS_CFG, MSG_CFG_NAVX5)

  def receive_message(self, ignore_eof=False):
    '''blocking receive of one ublox message'''
    msg = UBloxMessage()
    while True:
      n = msg.needed_bytes()
      b = self.read(n)
      if not b:
        if ignore_eof:
          time.sleep(0.01)
          continue
        if len(msg._buf) > 0:
          self.debug(1, "dropping %d bytes" % len(msg._buf))
        return None
      msg.add(b)
      if self.log is not None:
        self.log.write(b)
        self.log.flush()
      if msg.valid():
        self.special_handling(msg)
        return msg

  def receive_message_noerror(self, ignore_eof=False):
    '''blocking receive of one ublox message, ignoring errors'''
    try:
      return self.receive_message(ignore_eof=ignore_eof)
    except UBloxError as e:
      print(e)
      return None
    except OSError as e:
      # Occasionally we get hit with 'resource temporarily unavailable'
      # messages here on the serial device, catch them too.
      print(e)
      return None

  def send(self, msg):
    '''send a preformatted ublox message'''
    if not msg.valid():
      self.debug(1, "invalid send")
      return
    if not self.read_only:
      self.write(msg._buf)

  def send_message(self, msg_class, msg_id, payload):
    '''send a ublox message with class, id and payload'''
    msg = UBloxMessage()
    msg._buf = struct.pack('<BBBBH', 0xb5, 0x62, msg_class, msg_id, len(payload))
    msg._buf += payload
    (ck_a, ck_b) = msg.checksum(msg._buf[2:])
    msg._buf += struct.pack('<BB', ck_a, ck_b)
    self.send(msg)

  def configure_solution_rate(self, rate_ms=200, nav_rate=1, timeref=0):
    '''configure the solution rate in milliseconds'''
    payload = struct.pack('<HHH', rate_ms, nav_rate, timeref)
    self.send_message(CLASS_CFG, MSG_CFG_RATE, payload)

  def configure_message_rate(self, msg_class, msg_id, rate):
    '''configure the message rate for a given message'''
    payload = struct.pack('<BBB', msg_class, msg_id, rate)
    self.send_message(CLASS_CFG, MSG_CFG_SET_RATE, payload)

  def configure_port(self, port=1, inMask=3, outMask=3, mode=2240, baudrate=None):
    '''configure a IO port'''
    if baudrate is None:
      baudrate = self.baudrate
    payload = struct.pack('<BBH8BHHBBBB', port, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, inMask,
                          outMask, 0, 0, 0, 0)
    self.send_message(CLASS_CFG, MSG_CFG_PRT, payload)

  def configure_loadsave(self, clearMask=0, saveMask=0, loadMask=0, deviceMask=0):
    '''configure configuration load/save'''
    payload = struct.pack('<IIIB', clearMask, saveMask, loadMask, deviceMask)
    self.send_message(CLASS_CFG, MSG_CFG_CFG, payload)

  def configure_poll(self, msg_class, msg_id, payload=b''):
    '''poll a configuration message'''
    self.send_message(msg_class, msg_id, payload)

  def configure_poll_port(self, portID=None):
    '''poll a port configuration'''
    if portID is None:
      self.configure_poll(CLASS_CFG, MSG_CFG_PRT)
    else:
      self.configure_poll(CLASS_CFG, MSG_CFG_PRT, struct.pack('<B', portID))

  def configure_min_max_sats(self, min_sats=4, max_sats=32):
    '''Set the minimum/maximum number of satellites for a solution in the NAVX5 message'''
    payload = struct.pack('<HHIBBBBBBBBBBHIBBBBBBHII', 0, 4, 0, 0, 0, min_sats, max_sats,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    self.send_message(CLASS_CFG, MSG_CFG_NAVX5, payload)

  def module_reset(self, set, mode):
    ''' Reset the module for hot/warm/cold start'''
    payload = struct.pack('<HBB', set, mode, 0)
    self.send_message(CLASS_CFG, MSG_CFG_RST, payload)
