# pylint: skip-file
from openpilot.common.transformations.orientation import numpy_wrap
from openpilot.common.transformations.transformations import (ecef2geodetic_single,
                                                    geodetic2ecef_single)
from openpilot.common.transformations.transformations import LocalCoord as LocalCoord_single


class LocalCoord(LocalCoord_single):
  ecef2ned = numpy_wrap(LocalCoord_single.ecef2ned_single, (3,), (3,))
  ned2ecef = numpy_wrap(LocalCoord_single.ned2ecef_single, (3,), (3,))
  geodetic2ned = numpy_wrap(LocalCoord_single.geodetic2ned_single, (3,), (3,))
  ned2geodetic = numpy_wrap(LocalCoord_single.ned2geodetic_single, (3,), (3,))


geodetic2ecef = numpy_wrap(geodetic2ecef_single, (3,), (3,))
ecef2geodetic = numpy_wrap(ecef2geodetic_single, (3,), (3,))

geodetic_from_ecef = ecef2geodetic
ecef_from_geodetic = geodetic2ecef
