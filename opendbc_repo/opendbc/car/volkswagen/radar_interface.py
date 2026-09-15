from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.volkswagen.values import DBC, VolkswagenFlags, CanBus

NO_OBJECT_ID = 0
LANE_TYPES = ("Same_Lane", "Left_Lane", "Right_Lane")
SIGNAL_SETS = tuple(
  (
    f"{prefix}_ObjectID",
    f"{prefix}_Long_Distance",
    f"{prefix}_Lat_Distance",
    f"{prefix}_Rel_Velo",
  )
  for lane in LANE_TYPES
  for idx in (1, 2)
  for prefix in (f"{lane}_0{idx}",)
)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)

    # With the MEB gateway harness, we do not have access to the raw points from the radar.
    # However, the camera publishes decent, albeit filtered, tracks. Two for each lane; left, center, and right.
    self.rcp: CANParser | None = None
    if CP.flags & VolkswagenFlags.MEB and not self.CP.radarUnavailable:
      self.rcp = CANParser(DBC[CP.carFingerprint][Bus.radar], [("MEB_Distance_01", 25)], CanBus(CP).cam)

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    self.rcp.update(can_strings)

    if len(self.rcp.vl_all["MEB_Distance_01"]["Distance_Status"]) == 0:
      return None

    return self._update()

  def _update(self):
    ret = structs.RadarData()

    if not self.rcp.can_valid:
      ret.errors.canError = True
      return ret

    msg = self.rcp.vl["MEB_Distance_01"]

    # Can be 3 when radar sensor is obstructed
    if msg["Distance_Status"] != 0:
      ret.errors.radarUnavailableTemporary = True

    seen_ids = set()
    for obj_id_sig, long_sig, lat_sig, vel_sig in SIGNAL_SETS:
      obj_id = int(msg[obj_id_sig])
      if obj_id == NO_OBJECT_ID:
        continue

      # We shouldn't see duplicate track ids
      if obj_id in seen_ids:
        ret.errors.radarFault = True
        return ret

      seen_ids.add(obj_id)

      if obj_id not in self.pts:
        pt = structs.RadarData.RadarPoint()
        pt.trackId = self.track_id
        self.track_id += 1
        self.pts[obj_id] = pt
      else:
        pt = self.pts[obj_id]

      pt.dRel = msg[long_sig]
      pt.yRel = msg[lat_sig]
      pt.vRel = msg[vel_sig]

    inactive_ids = self.pts.keys() - seen_ids
    for obj_id in inactive_ids:
      self.pts.pop(obj_id, None)

    ret.points = list(self.pts.values())
    return ret
