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
from selfdrive.car.nissan.values import CAR as NISSAN


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
  "420a8e183f1aed48|2020-03-05--07-15-29": {
    'carFingerprint': CHRYSLER.PACIFICA_2017_HYBRID,
    'enableCamera': True,
  },
  "8190c7275a24557b|2020-01-29--08-33-58": {  # 2020 model year
    'carFingerprint': CHRYSLER.PACIFICA_2019_HYBRID,
    'enableCamera': True,
  },
  # This pacifica was removed because the fingerprint seemed from a Volt
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
  "c950e28c26b5b168|2018-05-30--22-03-41": {
    'carFingerprint': GM.VOLT,
    'enableCamera': True,
  },
  # TODO: use another route that has radar data at start
  "7cc2a8365b4dd8a9|2018-12-02--12-10-44": {
    'carFingerprint': GM.ACADIA,
    'enableCamera': True,
  },
  "aa20e335f61ba898|2019-02-05--16-59-04": {
    'carFingerprint': GM.BUICK_REGAL,
    'enableCamera': True,
  },
  "7d44af5b7a1b2c8e|2017-09-16--01-50-07": {
    'carFingerprint': HONDA.CIVIC,
    'enableCamera': True,
  },
  "a859a044a447c2b0|2020-03-03--18-42-45": {
    'carFingerprint': HONDA.CRV_EU,
    'enableCamera': True,
    'fingerprintSource': 'fixed',
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
  "81722949a62ea724|2019-03-29--15-51-26": {
    'carFingerprint': HONDA.ODYSSEY_CHN,
    'enableCamera': False,
  },
  "81722949a62ea724|2019-04-06--15-19-25": {
    'carFingerprint': HONDA.ODYSSEY_CHN,
    'enableCamera': True,
  },
  "08a3deb07573f157|2020-03-06--16-11-19": {
    'carFingerprint': HONDA.ACCORD_15,
    'enableCamera': True,
  },
  "f1b4c567731f4a1b|2018-06-06--14-43-46": {
    'carFingerprint': HONDA.ACCORD,
    'enableCamera': True,
  },
  "690c4c9f9f2354c7|2018-09-15--17-36-05": {
    'carFingerprint': HONDA.ACCORDH,
    'enableCamera': True,
  },
  "1ad763dd22ef1a0e|2020-02-29--18-37-03": {
    'carFingerprint': HONDA.CRV_5G,
    'enableCamera': True,
  },
  "0a96f86fcfe35964|2020-02-05--07-25-51": {
    'carFingerprint': HONDA.ODYSSEY,
    'enableCamera': True,
  },
  "d83f36766f8012a5|2020-02-05--18-42-21": {
    'carFingerprint': HONDA.CIVIC_BOSCH_DIESEL,
    'enableCamera': True,
    'fingerprintSource': 'fixed',
  },
  "fb51d190ddfd8a90|2020-02-25--14-43-43": {
    'carFingerprint': HONDA.INSIGHT,
    'enableCamera': True,
    'fingerprintSource': 'fixed',
  },
  "07d37d27996096b6|2020-03-04--21-57-27": {
    'carFingerprint': HONDA.PILOT,
    'enableCamera': True,
  },
  "22affd6c545d985e|2020-03-08--01-08-09": {
    'carFingerprint': HONDA.PILOT_2019,
    'enableCamera': True,
  },
  "0a78dfbacc8504ef|2020-03-04--13-29-55": {
    'carFingerprint': HONDA.CIVIC_BOSCH,
    'enableCamera': True,
  },
  "38bfd238edecbcd7|2018-08-22--09-45-44": {
    'carFingerprint': HYUNDAI.SANTA_FE,
    'enableCamera': False,
  },
  "38bfd238edecbcd7|2018-08-29--22-02-15": {
    'carFingerprint': HYUNDAI.SANTA_FE,
    'enableCamera': True,
  },
  "7653b2bce7bcfdaa|2020-03-04--15-34-32": {
    'carFingerprint': HYUNDAI.KIA_OPTIMA,
    'enableCamera': True,
  },
  "f7b6be73e3dfd36c|2019-05-12--18-07-16": {
    'carFingerprint': TOYOTA.AVALON,
    'enableCamera': False,
    'enableDsu': False,
  },
  "6cdecc4728d4af37|2020-02-23--15-44-18": {
    'carFingerprint': TOYOTA.CAMRY,
    'enableCamera': True,
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
  "32a7df20486b0f70|2020-02-06--16-06-50": {
    'carFingerprint': TOYOTA.RAV4H,
    'enableCamera': True,
    'enableDsu': True,
    'enableGasInterceptor': False,
  },
  "cdf2f7de565d40ae|2019-04-25--03-53-41": {
    'carFingerprint': TOYOTA.RAV4_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
    "7e34a988419b5307|2019-12-18--19-13-30": {
    'carFingerprint': TOYOTA.RAV4H_TSS2,
    'enableCamera': True,
    'fingerprintSource': 'fixed'
  },
  "e6a24be49a6cd46e|2019-10-29--10-52-42": {
    'carFingerprint': TOYOTA.LEXUS_ES_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "25057fa6a5a63dfb|2020-03-04--08-44-23": {
    'carFingerprint': TOYOTA.LEXUS_CTH,
    'enableCamera': True,
    'enableDsu': True,
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
  "ec429c0f37564e3c|2020-02-01--17-28-12": {
    'carFingerprint': TOYOTA.LEXUS_NXH,
    'enableCamera': True,
    'enableDsu': False,
  },
  # TODO: missing some combos for highlander
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
  "eb6acd681135480d|2019-06-20--20-00-00": {
    'carFingerprint': TOYOTA.SIENNA,
    'enableCamera': True,
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
  "fbbfa6af821552b9|2020-03-03--08-09-43": {
    'carFingerprint': NISSAN.XTRAIL,
    'enableCamera': True,
  },
  "5b7c365c50084530|2020-03-25--22-10-13": {
    'carFingerprint': NISSAN.LEAF,
    'enableCamera': True,
  },
}

passive_routes = [
]

forced_dashcam_routes = [
  # Ford fusion
  "f1b4c567731f4a1b|2018-04-18--11-29-37",
  "f1b4c567731f4a1b|2018-04-30--10-15-35",
]

# TODO: add routes for these cars
non_tested_cars = [
  CHRYSLER.JEEP_CHEROKEE,
  CHRYSLER.JEEP_CHEROKEE_2019,
  CHRYSLER.PACIFICA_2018,
  CHRYSLER.PACIFICA_2020,
  CHRYSLER.PACIFICA_2018_HYBRID,
  GM.CADILLAC_ATS,
  GM.HOLDEN_ASTRA,
  GM.MALIBU,
  HONDA.ACURA_RDX,
  HONDA.CRV,
  HONDA.RIDGELINE,
  HYUNDAI.ELANTRA,
  HYUNDAI.GENESIS,
  HYUNDAI.KIA_SORENTO,
  HYUNDAI.KIA_STINGER,
  TOYOTA.CAMRYH,
  TOYOTA.CHR,
  TOYOTA.CHRH,
  TOYOTA.HIGHLANDERH,
]

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
    print("GETTING ROUTE LOGS")
    get_route_log(route)
    print("DONE GETTING ROUTE LOGS")

    params = Params()
    params.clear_all()
    params.manager_start()
    params.put("OpenpilotEnabledToggle", "1")
    params.put("CommunityFeaturesToggle", "1")
    params.put("Passive", "1" if route in passive_routes else "0")

    if checks.get('fingerprintSource', None) == 'fixed':
      os.environ['FINGERPRINT'] = checks['carFingerprint']
    else:
      os.environ['FINGERPRINT'] = ""

    print("testing ", route, " ", checks['carFingerprint'])
    print("Starting processes")
    for p in tested_procs:
      manager.start_managed_process(p)

    # Start unlogger
    print("Start unlogger")
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
