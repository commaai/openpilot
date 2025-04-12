#!/usr/bin/env python3

from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.route import Route

TEST_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
route = Route(TEST_ROUTE)
segnum = 2

road_img = FrameReader(route.camera_paths()[segnum]).get(0, pix_fmt="nv12")
