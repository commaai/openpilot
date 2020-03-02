import json
import os
import sys

from tools.lib.route import Route

route_name = sys.argv[1]
routes = Route(route_name)
data_dump = {
    "camera":routes.camera_paths(),
    "logs":routes.log_paths()
}

json.dump(data_dump, open("routes.json", "w"))