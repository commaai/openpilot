#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from typing import List, cast

import requests

import cereal.messaging as messaging
import selfdrive.manager as manager
from cereal import car
from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.chrysler.values import CAR as CHRYSLER
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.ford.values import CAR as FORD
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.honda.values import CAR as HONDA
from selfdrive.car.hyundai.values import CAR as HYUNDAI
from selfdrive.car.nissan.values import CAR as NISSAN
from selfdrive.car.mazda.values import CAR as MAZDA
from selfdrive.car.subaru.values import CAR as SUBARU
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN

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
    except Exception:
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
  "0e7a2ba168465df5|2020-10-18--14-14-22": {
    'carFingerprint': HONDA.ACURA_RDX_3G,
    'enableCamera': True,
    'fingerprintSource': 'fixed',
  },
  "a74b011b32b51b56|2020-07-26--17-09-36": {
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
  "03be5f2fd5c508d1|2020-04-19--18-44-15": {
    'carFingerprint': HONDA.HRV,
    'enableCamera': True,
    'fingerprintSource': 'fixed',
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
  "f34a60d68d83b1e5|2020-10-06--14-35-55": {
    'carFingerprint': HONDA.ACURA_RDX,
    'enableCamera': True,
  },
  "6fe86b4e410e4c37|2020-07-22--16-27-13": {
    'carFingerprint': HYUNDAI.HYUNDAI_GENESIS,
    'enableCamera': True,
  },
  "70c5bec28ec8e345|2020-08-08--12-22-23": {
    'carFingerprint': HYUNDAI.GENESIS_G70,
    'enableCamera': True,
  },
  "6b301bf83f10aa90|2020-11-22--16-45-07": {
    'carFingerprint': HYUNDAI.GENESIS_G80,
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
  "5b7c365c50084530|2020-04-15--16-13-24": {
    'carFingerprint': HYUNDAI.SONATA,
    'enableCamera': True,
  },
  "b2a38c712dcf90bd|2020-05-18--18-12-48": {
    'carFingerprint': HYUNDAI.SONATA_2019,
    'enableCamera': True,
  },
  "5875672fc1d4bf57|2020-07-23--21-33-28": {
    'carFingerprint': HYUNDAI.KIA_SORENTO,
    'enableCamera': True,
  },
  "9c917ba0d42ffe78|2020-04-17--12-43-19": {
    'carFingerprint': HYUNDAI.PALISADE,
    'enableCamera': True,
  },
  "610ebb9faaad6b43|2020-06-13--15-28-36": {
    'carFingerprint': HYUNDAI.IONIQ_EV_LTD,
    'enableCamera': True,
  },
  "2c5cf2dd6102e5da|2020-06-26--16-00-08": {
    'carFingerprint': HYUNDAI.IONIQ,
    'enableCamera': True,
  },
  "22d955b2cd499c22|2020-08-10--19-58-21": {
    'carFingerprint': HYUNDAI.KONA,
    'enableCamera': True,
  },
  "5dddcbca6eb66c62|2020-07-26--13-24-19": {
    'carFingerprint': HYUNDAI.KIA_STINGER,
    'enableCamera': True,
  },
  "d624b3d19adce635|2020-08-01--14-59-12": {
    'carFingerprint': HYUNDAI.VELOSTER,
    'enableCamera': True,
  },
  "50c6c9b85fd1ff03|2020-10-26--17-56-06": {
    'carFingerprint': HYUNDAI.KIA_NIRO_EV,
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
    'fingerprintSource': 'fixed',
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
  "b14c5b4742e6fc85|2020-07-28--19-50-11": {
    'carFingerprint': TOYOTA.RAV4,
    'enableCamera': True,
    'enableDsu': False,
    'enableGasInterceptor': True,
  },
  "32a7df20486b0f70|2020-02-06--16-06-50": {
    'carFingerprint': TOYOTA.RAV4H,
    'enableCamera': True,
    'enableDsu': True,
    'enableGasInterceptor': False,
    'fingerprintSource': 'fixed',
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
    'fingerprintSource': 'fixed',
  },
  "b74758c690a49668|2020-05-20--15-58-57": {
    'carFingerprint': TOYOTA.LEXUS_RXH_TSS2,
    'enableCamera': True,
    'fingerprintSource': 'fixed',
  },
  "ec429c0f37564e3c|2020-02-01--17-28-12": {
    'carFingerprint': TOYOTA.LEXUS_NXH,
    'enableCamera': True,
    'enableDsu': False,
  },
  "964c09eb11ca8089|2020-11-03--22-04-00": {
    'carFingerprint': TOYOTA.LEXUS_NX,
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
  "0a0de17a1e6a2d15|2020-09-21--21-24-41": {
    'carFingerprint': TOYOTA.PRIUS_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "76b83eb0245de90e|2019-10-20--15-42-29": {
    'carFingerprint': VOLKSWAGEN.GOLF,
    'enableCamera': True,
  },
  "3c8f0c502e119c1c|2020-06-30--12-58-02": {
    'carFingerprint': SUBARU.ASCENT,
    'enableCamera': True,
  },
  "c321c6b697c5a5ff|2020-06-23--11-04-33": {
    'carFingerprint': SUBARU.FORESTER,
    'enableCamera': True,
  },
  "791340bc01ed993d|2019-03-10--16-28-08": {
    'carFingerprint': SUBARU.IMPREZA,
    'enableCamera': True,
  },
  # Dashcam
  "95441c38ae8c130e|2020-06-08--12-10-17": {
    'carFingerprint': SUBARU.FORESTER_PREGLOBAL,
    'enableCamera': True,
  },
  # Dashcam
  "df5ca7660000fba8|2020-06-16--17-37-19": {
    'carFingerprint': SUBARU.LEGACY_PREGLOBAL,
    'enableCamera': True,
  },
  # Dashcam
  "5ab784f361e19b78|2020-06-08--16-30-41": {
    'carFingerprint': SUBARU.OUTBACK_PREGLOBAL,
    'enableCamera': True,
  },
  # Dashcam
  "e19eb5d5353b1ac1|2020-08-09--14-37-56": {
    'carFingerprint': SUBARU.OUTBACK_PREGLOBAL_2018,
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
  "059ab9162e23198e|2020-05-30--09-41-01": {
    'carFingerprint': NISSAN.ROGUE,
    'enableCamera': True,
  },
  "32a319f057902bb3|2020-04-27--15-18-58": {
    'carFingerprint': MAZDA.CX5,
    'enableCamera': True,
  },
  "10b5a4b380434151|2020-08-26--17-11-45": {
    'carFingerprint': MAZDA.CX9,
    'enableCamera': True,
  },
  "74f1038827005090|2020-08-26--20-05-50": {
    'carFingerprint': MAZDA.Mazda3,
    'enableCamera': True,
  },
}

passive_routes: List[str] = [
]

forced_dashcam_routes = [
  # Ford fusion
  "f1b4c567731f4a1b|2018-04-18--11-29-37",
  "f1b4c567731f4a1b|2018-04-30--10-15-35",
  # Mazda CX5
  "32a319f057902bb3|2020-04-27--15-18-58",
  # Mazda CX9
  "10b5a4b380434151|2020-08-26--17-11-45",
  # Mazda3
  "74f1038827005090|2020-08-26--20-05-50",
]

# TODO: add routes for these cars
non_tested_cars = [
  CHRYSLER.JEEP_CHEROKEE,
  CHRYSLER.JEEP_CHEROKEE_2019,
  CHRYSLER.PACIFICA_2018,
  CHRYSLER.PACIFICA_2018_HYBRID,
  CHRYSLER.PACIFICA_2020,
  GM.CADILLAC_ATS,
  GM.HOLDEN_ASTRA,
  GM.MALIBU,
  HONDA.CRV,
  HONDA.RIDGELINE,
  HYUNDAI.ELANTRA,
  HYUNDAI.ELANTRA_GT_I30,
  HYUNDAI.GENESIS_G90,
  HYUNDAI.KIA_FORTE,
  HYUNDAI.KIA_OPTIMA_H,
  HYUNDAI.KONA_EV,
  TOYOTA.CAMRYH,
  TOYOTA.CHR,
  TOYOTA.CHRH,
  TOYOTA.HIGHLANDER,
  TOYOTA.HIGHLANDERH,
  TOYOTA.HIGHLANDERH_TSS2,
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
    get_route_log(route)

    params = Params()
    params.clear_all()
    params.manager_start()
    params.put("OpenpilotEnabledToggle", "1")
    params.put("CommunityFeaturesToggle", "1")
    params.put("Passive", "1" if route in passive_routes else "0")

    os.environ['SKIP_FW_QUERY'] = "1"
    if checks.get('fingerprintSource', None) == 'fixed':
      os.environ['FINGERPRINT'] = cast(str, checks['carFingerprint'])
    else:
      os.environ['FINGERPRINT'] = ""

    print("testing ", route, " ", checks['carFingerprint'])
    print("Starting processes")
    for p in tested_procs:
      manager.start_managed_process(p)

    # Start unlogger
    print("Start unlogger")
    unlogger_cmd = [os.path.join(BASEDIR, 'tools/replay/unlogger.py'), route, '/tmp']
    disable_socks = 'frame,encodeIdx,plan,pathPlan,liveLongitudinalMpc,radarState,controlsState,liveTracks,liveMpc,sendcan,carState,carControl,carEvents,carParams'
    unlogger = subprocess.Popen(unlogger_cmd + ['--disable', disable_socks, '--no-interactive'], preexec_fn=os.setsid)  # pylint: disable=subprocess-popen-preexec-fn

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
