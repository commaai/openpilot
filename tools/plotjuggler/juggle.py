#!/usr/bin/python3

import sys
import subprocess
import tempfile
from tools.lib.route import Route
from tools.lib.url_file import URLFile

# for now just /0 segment
# route_name = str(sys.argv[1]).replace('|', '/')
route_name = "0982d79ebb0de295|2021-01-17--17-13-08"

r = Route(route_name)
lp = r.log_paths()[0]

uf = URLFile(lp)

subprocess.call(f"~/PlotJuggler/build/bin/plotjuggler -d {uf.name}", shell=True)


