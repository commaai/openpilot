#!/usr/bin/env python3
import time
import os
import sys
import signal
import subprocess
import requests
from cereal import car

import selfdrive.manager as manager
import cereal.messaging as messaging
from common.params import Params
from common.basedir import BASEDIR
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.honda.values import CAR as HONDA
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.ford.values import CAR as FORD
from selfdrive.car.hyundai.values import CAR as HYUNDAI
from selfdrive.car.chrysler.values import CAR as CHRYSLER
from selfdrive.car.subaru.values import CAR as SUBARU
from selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN
from selfdrive.car.mock.values import CAR as MOCK


os.environ['NOCRASH'] = '1'


def wait_for_sockets(socks, timeout=10.0):
  sm = messaging.SubMaster(socks)
  t = time.time()

  recvd = []
  while time.time() - t < timeout and len(recvd) < len(socks):
    sm.update()
    for s in socks:
      if s not in recvd and sm.updated[s]:
        recvd.append(s)
  return recvd

def get_route_log(route_name):
  log_path = os.path.join("/tmp", "%s--0--%s" % (route_name.replace("|", "_"), "rlog.bz2"))

  if not os.path.isfile(log_path):
    log_url = "https://commadataci.blob.core.windows.net/openpilotci/%s/0/%s" % (route_name.replace("|", "/"), "rlog.bz2")

    # if request fails, try again once and let it throw exception if fails again
    try:
      r = requests.get(log_url, timeout=15)
    except:
      r = requests.get(log_url, timeout=15)

    if r.status_code == 200:
      with open(log_path, "wb") as f:
        f.write(r.content)
    else:
      print("failed to download test log %s" % route_name)
      sys.exit(-1)

routes = {

  "975b26878285314d|2018-12-25--14-42-13": {
    'carFingerprint': CHRYSLER.PACIFICA_2018_HYBRID,
    'enableCamera': True,
  },
  "b0c9d2329ad1606b|2019-01-06--10-11-23": {
    'carFingerprint': CHRYSLER.PACIFICA_2017_HYBRID,
    'enableCamera': True,
  },
  "0607d2516fc2148f|2019-02-13--23-03-16": {
    'carFingerprint': CHRYSLER.PACIFICA_2019_HYBRID,
    'enableCamera': True,
  },
  "8190c7275a24557b|2020-01-29--08-33-58": {
    'carFingerprint': CHRYSLER.PACIFICA_2020_HYBRID,
    'enableCamera': True,
  },
  # This pacifica was removed because the fingerprint seemed from a Volt
  #"9f7a7e50a51fb9db|2019-01-03--14-05-01": {
  #  'carFingerprint': CHRYSLER.PACIFICA_2018,
  #  'enableCamera': True,
  #},
  "9f7a7e50a51fb9db|2019-01-17--18-34-21": {
    'carFingerprint': CHRYSLER.JEEP_CHEROKEE,
    'enableCamera': True,
  },
  "192a598e34926b1e|2019-04-04--13-27-39": {
    'carFingerprint': CHRYSLER.JEEP_CHEROKEE_2019,
    'enableCamera': True,
  },
  "f1b4c567731f4a1b|2018-04-18--11-29-37": {
    'carFingerprint': FORD.FUSION,
    'enableCamera': False,
  },
  "f1b4c567731f4a1b|2018-04-30--10-15-35": {
    'carFingerprint': FORD.FUSION,
    'enableCamera': True,
  },
  "7ed9cdf8d0c5f43e|2018-05-17--09-31-36": {
    'carFingerprint': GM.CADILLAC_CT6,
    'enableCamera': True,
  },
  "265007255e794bce|2018-11-24--22-08-31": {
    'carFingerprint': GM.CADILLAC_ATS,
    'enableCamera': True,
  },
  "c950e28c26b5b168|2018-05-30--22-03-41": {
    'carFingerprint': GM.VOLT,
    'enableCamera': True,
  },
  # TODO: use another route that has radar data at start
  "aadda448b49c99ad|2018-10-25--17-16-22": {
    'carFingerprint': GM.MALIBU,
    'enableCamera': True,
  },
  "49c73650e65ff465|2018-11-19--16-58-04": {
    'carFingerprint': GM.HOLDEN_ASTRA,
    'enableCamera': True,
  },
  "7cc2a8365b4dd8a9|2018-12-02--12-10-44": {
    'carFingerprint': GM.ACADIA,
    'enableCamera': True,
  },
  "aa20e335f61ba898|2018-12-17--21-10-37": {
    'carFingerprint': GM.BUICK_REGAL,
    'enableCamera': False,
  },
  "aa20e335f61ba898|2019-02-05--16-59-04": {
    'carFingerprint': GM.BUICK_REGAL,
    'enableCamera': True,
  },
  "7d44af5b7a1b2c8e|2017-09-16--01-50-07": {
    'carFingerprint': HONDA.CIVIC,
    'enableCamera': True,
  },
  "c9d60e5e02c04c5c|2018-01-08--16-01-49": {
    'carFingerprint': HONDA.CRV,
    'enableCamera': True,
  },
  "1851183c395ef471|2018-05-31--18-07-21": {
    'carFingerprint': HONDA.CRV_5G,
    'enableCamera': True,
  },
  "232585b7784c1af4|2019-04-08--14-12-14": {
    'carFingerprint': HONDA.CRV_HYBRID,
    'enableCamera': True,
  },
  "99e3eaed7396619e|2019-08-13--15-07-03": {
    'carFingerprint': HONDA.FIT,
    'enableCamera': True,
  },
  "2ac95059f70d76eb|2018-02-05--15-03-29": {
    'carFingerprint': HONDA.ACURA_ILX,
    'enableCamera': True,
  },
  "21aa231dee2a68bd|2018-01-30--04-54-41": {
    'carFingerprint': HONDA.ODYSSEY,
    'enableCamera': True,
  },
  "81722949a62ea724|2019-03-29--15-51-26": {
    'carFingerprint': HONDA.ODYSSEY_CHN,
    'enableCamera': False,
  },
  "81722949a62ea724|2019-04-06--15-19-25": {
    'carFingerprint': HONDA.ODYSSEY_CHN,
    'enableCamera': True,
  },
  "5a2cfe4bb362af9e|2018-02-02--23-41-07": {
    'carFingerprint': HONDA.ACURA_RDX,
    'enableCamera': True,
  },
  "3e9592a1c78a3d63|2018-02-08--20-28-24": {
    'carFingerprint': HONDA.PILOT,
    'enableCamera': True,
  },
  "34a84d2b765df688|2018-08-28--12-41-00": {
    'carFingerprint': HONDA.PILOT_2019,
    'enableCamera': True,
  },
  "900ad17e536c3dc7|2018-04-12--22-02-36": {
    'carFingerprint': HONDA.RIDGELINE,
    'enableCamera': True,
  },
  "f1b4c567731f4a1b|2018-06-06--14-43-46": {
    'carFingerprint': HONDA.ACCORD,
    'enableCamera': True,
  },
  "1582e1dc57175194|2018-08-15--07-46-07": {
    'carFingerprint': HONDA.ACCORD_15,
    'enableCamera': True,
  },
  # TODO: This doesnt fingerprint because the fingerprint overlaps with the Insight
  # "690c4c9f9f2354c7|2018-09-15--17-36-05": {
  #   'carFingerprint': HONDA.ACCORDH,
  #   'enableCamera': True,
  # },
  "1632088eda5e6c4d|2018-06-07--08-03-18": {
    'carFingerprint': HONDA.CIVIC_BOSCH,
    'enableCamera': True,
  },
  #"18971a99f3f2b385|2018-11-14--19-09-31": {
  #  'carFingerprint': HONDA.INSIGHT,
  #  'enableCamera': True,
  #},
  "38bfd238edecbcd7|2018-08-22--09-45-44": {
    'carFingerprint': HYUNDAI.SANTA_FE,
    'enableCamera': False,
  },
  "38bfd238edecbcd7|2018-08-29--22-02-15": {
    'carFingerprint': HYUNDAI.SANTA_FE,
    'enableCamera': True,
  },
  "a893a80e5c5f72c8|2019-01-14--20-02-59": {
    'carFingerprint': HYUNDAI.GENESIS,
    'enableCamera': True,
  },
  "9d5fb4f0baa1b3e1|2019-01-14--17-45-59": {
    'carFingerprint': HYUNDAI.KIA_SORENTO,
    'enableCamera': True,
  },
  "215cd70e9c349266|2018-11-25--22-22-12": {
    'carFingerprint': HYUNDAI.KIA_STINGER,
    'enableCamera': True,
  },
  "31390e3eb6f7c29a|2019-01-23--08-56-00": {
    'carFingerprint': HYUNDAI.KIA_OPTIMA,
    'enableCamera': True,
  },
  "53ac3251e03f95d7|2019-01-10--13-43-32": {
    'carFingerprint': HYUNDAI.ELANTRA,
    'enableCamera': True,
  },
  "f7b6be73e3dfd36c|2019-05-12--18-07-16": {
    'carFingerprint': TOYOTA.AVALON,
    'enableCamera': False,
    'enableDsu': False,
  },
  "f7b6be73e3dfd36c|2019-05-11--22-34-20": {
    'carFingerprint': TOYOTA.AVALON,
    'enableCamera': True,
    'enableDsu': False,
  },
  "b0f5a01cf604185c|2018-01-26--00-54-32": {
    'carFingerprint': TOYOTA.COROLLA,
    'enableCamera': True,
    'enableDsu': True,
  },
  "b0f5a01cf604185c|2018-01-26--10-54-38": {
    'carFingerprint': TOYOTA.COROLLA,
    'enableCamera': True,
    'enableDsu': False,
  },
  "b0f5a01cf604185c|2018-01-26--10-59-31": {
    'carFingerprint': TOYOTA.COROLLA,
    'enableCamera': False,
    'enableDsu': False,
  },
  "5f5afb36036506e4|2019-05-14--02-09-54": {
    'carFingerprint': TOYOTA.COROLLA_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "5ceff72287a5c86c|2019-10-19--10-59-02": {
    'carFingerprint': TOYOTA.COROLLAH_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "56fb1c86a9a86404|2017-11-10--10-18-43": {
    'carFingerprint': TOYOTA.PRIUS,
    'enableCamera': True,
    'enableDsu': True,
  },
  "b0f5a01cf604185c|2017-12-18--20-32-32": {
    'carFingerprint': TOYOTA.RAV4,
    'enableCamera': True,
    'enableDsu': True,
    'enableGasInterceptor': False,
  },
  "b0c9d2329ad1606b|2019-04-02--13-24-43": {
    'carFingerprint': TOYOTA.RAV4,
    'enableCamera': True,
    'enableDsu': True,
    'enableGasInterceptor': True,
  },
  "cdf2f7de565d40ae|2019-04-25--03-53-41": {
    'carFingerprint': TOYOTA.RAV4_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
    "7e34a988419b5307|2019-12-18--19-13-30": {
    'carFingerprint': TOYOTA.RAV4H_TSS2,
    'enableCamera': True,
  },
  "e6a24be49a6cd46e|2019-10-29--10-52-42": {
    'carFingerprint': TOYOTA.LEXUS_ES_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "f49e8041283f2939|2019-05-29--13-48-33": {
    'carFingerprint': TOYOTA.LEXUS_ESH_TSS2,
    'enableCamera': False,
    'enableDsu': False,
  },
  "f49e8041283f2939|2019-05-30--11-51-51": {
    'carFingerprint': TOYOTA.LEXUS_ESH_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
    "886fcd8408d570e9|2020-01-29--05-11-22": {
      'carFingerprint': TOYOTA.LEXUS_RX,
      'enableCamera': True,
      'enableDsu': True,
    },
    "886fcd8408d570e9|2020-01-29--02-18-55": {
      'carFingerprint': TOYOTA.LEXUS_RX,
      'enableCamera': True,
      'enableDsu': False,
    },
  "b0f5a01cf604185c|2018-02-01--21-12-28": {
    'carFingerprint': TOYOTA.LEXUS_RXH,
    'enableCamera': True,
    'enableDsu': True,
  },
    "01b22eb2ed121565|2020-02-02--11-25-51": {
    'carFingerprint': TOYOTA.LEXUS_RX_TSS2,
    'enableCamera': True,
  },
  #FIXME: This works sometimes locally, but never in CI. Timing issue?
  #"b0f5a01cf604185c|2018-01-31--20-11-39": {
  #  'carFingerprint': TOYOTA.LEXUS_RXH,
  #  'enableCamera': False,
  #  'enableDsu': False,
  #},
  "8ae193ceb56a0efe|2018-06-18--20-07-32": {
    'carFingerprint': TOYOTA.RAV4H,
    'enableCamera': True,
    'enableDsu': True,
  },
  "fd10b9a107bb2e49|2018-07-24--16-32-42": {
    'carFingerprint': TOYOTA.CHR,
    'enableCamera': True,
    'enableDsu': False,
  },
  "fd10b9a107bb2e49|2018-07-24--20-32-08": {
    'carFingerprint': TOYOTA.CHR,
    'enableCamera': False,
    'enableDsu': False,
  },
  "b4c18bf13d5955da|2018-07-29--13-39-46": {
    'carFingerprint': TOYOTA.CHRH,
    'enableCamera': True,
    'enableDsu': False,
  },
  "d2d8152227f7cb82|2018-07-25--13-40-56": {
    'carFingerprint': TOYOTA.CAMRY,
    'enableCamera': True,
    'enableDsu': False,
  },
  "fbd011384db5e669|2018-07-26--20-51-48": {
    'carFingerprint': TOYOTA.CAMRYH,
    'enableCamera': True,
    'enableDsu': False,
  },
  # TODO: This replay has no good model/video
  # "c9fa2dd0f76caf23|2018-02-10--13-40-28": {
  #   'carFingerprint': TOYOTA.CAMRYH,
  #   'enableCamera': False,
  #   'enableDsu': False,
  # },
  # TODO: missingsome combos for highlander
  "0a302ffddbb3e3d3|2020-02-08--16-19-08": {
    'carFingerprint': TOYOTA.HIGHLANDER_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "aa659debdd1a7b54|2018-08-31--11-12-01": {
    'carFingerprint': TOYOTA.HIGHLANDER,
    'enableCamera': False,
    'enableDsu': False,
  },
  "362d23d4d5bea2fa|2018-09-02--17-03-55": {
    'carFingerprint': TOYOTA.HIGHLANDERH,
    'enableCamera': True,
    'enableDsu': True,
  },
  "eb6acd681135480d|2019-06-20--20-00-00": {
    'carFingerprint': TOYOTA.SIENNA,
    'enableCamera': True,
    'enableDsu': False,
  },
  "362d23d4d5bea2fa|2018-08-10--13-31-40": {
    'carFingerprint': TOYOTA.HIGHLANDERH,
    'enableCamera': False,
    'enableDsu': False,
  },
  "2e07163a1ba9a780|2019-08-25--13-15-13": {
    'carFingerprint': TOYOTA.LEXUS_IS,
    'enableCamera': True,
    'enableDsu': False,
  },
  "2e07163a1ba9a780|2019-08-29--09-35-42": {
    'carFingerprint': TOYOTA.LEXUS_IS,
    'enableCamera': False,
    'enableDsu': False,
  },
  "1dd19ceed0ee2b48|2018-12-22--17-36-49": {
    'carFingerprint': TOYOTA.LEXUS_IS, # 300 hybrid
    'enableCamera': True,
    'enableDsu': False,
  },
  "76b83eb0245de90e|2019-10-20--15-42-29": {
    'carFingerprint': VOLKSWAGEN.GOLF,
    'enableCamera': True,
  },
  "791340bc01ed993d|2019-03-10--16-28-08": {
    'carFingerprint': SUBARU.IMPREZA,
    'enableCamera': True,
  },
  # Tesla route, should result in mock car
  "07cb8a788c31f645|2018-06-17--18-50-29": {
    'carFingerprint': MOCK.MOCK,
  },
  ## Route with no can data, should result in mock car. This is not supported anymore
  #"bfa17080b080f3ec|2018-06-28--23-27-47": {
  #  'carFingerprint': MOCK.MOCK,
  #},
}

passive_routes = [
  "07cb8a788c31f645|2018-06-17--18-50-29",
  #"bfa17080b080f3ec|2018-06-28--23-27-47",
]

forced_dashcam_routes = [
  # Ford fusion
  "f1b4c567731f4a1b|2018-04-18--11-29-37",
  "f1b4c567731f4a1b|2018-04-30--10-15-35",
]

# TODO: replace all these with public routes
non_public_routes = [
  "0607d2516fc2148f|2019-02-13--23-03-16",  # CHRYSLER PACIFICA HYBRID 2019
  "3e9592a1c78a3d63|2018-02-08--20-28-24",  # HONDA PILOT 2017 TOURING
  "aa20e335f61ba898|2019-02-05--16-59-04",  # BUICK REGAL ESSENCE 2018
  "1851183c395ef471|2018-05-31--18-07-21",  # HONDA CR-V 2017 EX
  "9d5fb4f0baa1b3e1|2019-01-14--17-45-59",  # KIA SORENTO GT LINE 2018
  "b4c18bf13d5955da|2018-07-29--13-39-46",  # TOYOTA C-HR HYBRID 2018
  "5a2cfe4bb362af9e|2018-02-02--23-41-07",  # ACURA RDX 2018 ACURAWATCH PLUS
  "362d23d4d5bea2fa|2018-08-10--13-31-40",  # TOYOTA HIGHLANDER HYBRID 2018
  "aa20e335f61ba898|2018-12-17--21-10-37",  # BUICK REGAL ESSENCE 2018
  "215cd70e9c349266|2018-11-25--22-22-12",  # KIA STINGER GT2 2018
  "192a598e34926b1e|2019-04-04--13-27-39",  # JEEP GRAND CHEROKEE 2019
  "34a84d2b765df688|2018-08-28--12-41-00",  # HONDA PILOT 2019 ELITE
  "b0c9d2329ad1606b|2019-01-06--10-11-23",  # CHRYSLER PACIFICA HYBRID 2017
  "31390e3eb6f7c29a|2019-01-23--08-56-00",  # KIA OPTIMA SX 2019
  "fd10b9a107bb2e49|2018-07-24--16-32-42",  # TOYOTA C-HR 2018
  "9f7a7e50a51fb9db|2019-01-17--18-34-21",  # JEEP GRAND CHEROKEE V6 2018
  "aadda448b49c99ad|2018-10-25--17-16-22",  # CHEVROLET MALIBU PREMIER 2017
  "362d23d4d5bea2fa|2018-09-02--17-03-55",  # TOYOTA HIGHLANDER HYBRID 2018
  "1582e1dc57175194|2018-08-15--07-46-07",  # HONDA ACCORD 2018 LX 1.5T
  "fd10b9a107bb2e49|2018-07-24--20-32-08",  # TOYOTA C-HR 2018
  "265007255e794bce|2018-11-24--22-08-31",  # CADILLAC ATS Premium Performance 2018
  "53ac3251e03f95d7|2019-01-10--13-43-32",  # HYUNDAI ELANTRA LIMITED ULTIMATE 2017
  "21aa231dee2a68bd|2018-01-30--04-54-41",  # HONDA ODYSSEY 2018 EX-L
  "900ad17e536c3dc7|2018-04-12--22-02-36",  # HONDA RIDGELINE 2017 BLACK EDITION
  "975b26878285314d|2018-12-25--14-42-13",  # CHRYSLER PACIFICA HYBRID 2018
  "8ae193ceb56a0efe|2018-06-18--20-07-32",  # TOYOTA RAV4 HYBRID 2017
  "a893a80e5c5f72c8|2019-01-14--20-02-59",  # HYUNDAI GENESIS 2018
  "49c73650e65ff465|2018-11-19--16-58-04",  # HOLDEN ASTRA RS-V BK 2017
  "d2d8152227f7cb82|2018-07-25--13-40-56",  # TOYOTA CAMRY 2018
  "07cb8a788c31f645|2018-06-17--18-50-29",  # mock
  "c9d60e5e02c04c5c|2018-01-08--16-01-49",  # HONDA CR-V 2016 TOURING
  "1632088eda5e6c4d|2018-06-07--08-03-18",  # HONDA CIVIC HATCHBACK 2017 SEDAN/COUPE 2019
  "fbd011384db5e669|2018-07-26--20-51-48",  # TOYOTA CAMRY HYBRID 2018
]

# TODO: add routes for these cars
non_tested_cars = [TOYOTA.LEXUS_CTH, CHRYSLER.PACIFICA_2018, HONDA.ACCORDH]

if __name__ == "__main__":

  tested_procs = ["controlsd", "radard", "plannerd"]
  tested_socks = ["radarState", "controlsState", "carState", "plan"]

  tested_cars = [keys["carFingerprint"] for route, keys in routes.items()]
  for car_model in all_known_cars():
    if car_model not in tested_cars:
      print("***** WARNING: %s not tested *****" % car_model)

      # TODO: skip these for now, but make sure any new ports get routes
      if car_model not in non_tested_cars:
        print("TEST FAILED: Missing route for car '%s'" % car_model)
        sys.exit(1)

  print("Preparing processes")
  for p in tested_procs:
    manager.prepare_managed_process(p)

  results = {}
  for route, checks in routes.items():
    if route not in non_public_routes:
      print("GETTING ROUTE LOGS")
      get_route_log(route)
      print("DONE GETTING ROUTE LOGS")
    elif "UNLOGGER_PATH" not in os.environ:
      continue

    params = Params()
    params.clear_all()
    params.manager_start()
    params.put("OpenpilotEnabledToggle", "1")
    params.put("CommunityFeaturesToggle", "1")
    params.put("Passive", "1" if route in passive_routes else "0")

    print("testing ", route, " ", checks['carFingerprint'])
    print("Starting processes")
    for p in tested_procs:
      manager.start_managed_process(p)

    # Start unlogger
    print("Start unlogger")
    if route in non_public_routes:
      unlogger_cmd = [os.path.join(BASEDIR, os.environ['UNLOGGER_PATH']), route]
    else:
      unlogger_cmd = [os.path.join(BASEDIR, 'tools/replay/unlogger.py'), route, '/tmp']
    unlogger = subprocess.Popen(unlogger_cmd + ['--disable', 'frame,encodeIdx,plan,pathPlan,liveLongitudinalMpc,radarState,controlsState,liveTracks,liveMpc,sendcan,carState,carControl,carEvents,carParams', '--no-interactive'], preexec_fn=os.setsid)

    print("Check sockets")
    extra_socks = []
    has_camera = checks.get('enableCamera', False)
    if (route not in passive_routes) and (route not in forced_dashcam_routes) and has_camera:
      extra_socks.append("sendcan")
    if route not in passive_routes:
      extra_socks.append("pathPlan")

    recvd_socks = wait_for_sockets(tested_socks + extra_socks, timeout=30)
    failures = [s for s in tested_socks + extra_socks if s not in recvd_socks]

    print("Check if everything is running")
    running = manager.get_running()
    for p in tested_procs:
      if not running[p].is_alive:
        failures.append(p)
      manager.kill_managed_process(p)
    os.killpg(os.getpgid(unlogger.pid), signal.SIGTERM)

    sockets_ok = len(failures) == 0
    params_ok = True

    try:
      car_params = car.CarParams.from_bytes(params.get("CarParams"))
      for k, v in checks.items():
        if not v == getattr(car_params, k):
          params_ok = False
          failures.append(k)
    except Exception:
      params_ok = False

    if sockets_ok and params_ok:
      print("Success")
      results[route] = True, failures
    else:
      print("Failure")
      results[route] = False, failures
      break

  # put back not passive to not leave the params in an unintended state
  Params().put("Passive", "0")

  for route in results:
    print(results[route])

  if not all(passed for passed, _ in results.values()):
    print("TEST FAILED")
    sys.exit(1)
  else:
    print("TEST SUCCESSFUL")
