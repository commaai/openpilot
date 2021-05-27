#!/usr/bin/env python3
from selfdrive.car.chrysler.values import CAR as CHRYSLER
from selfdrive.car.ford.values import CAR as FORD
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.honda.values import CAR as HONDA
from selfdrive.car.hyundai.values import CAR as HYUNDAI
from selfdrive.car.nissan.values import CAR as NISSAN
from selfdrive.car.mazda.values import CAR as MAZDA
from selfdrive.car.subaru.values import CAR as SUBARU
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN

# TODO: add routes for these cars
non_tested_cars = [
  CHRYSLER.JEEP_CHEROKEE_2019,
  CHRYSLER.PACIFICA_2018,
  CHRYSLER.PACIFICA_2018_HYBRID,
  CHRYSLER.PACIFICA_2020,
  GM.CADILLAC_ATS,
  GM.HOLDEN_ASTRA,
  GM.MALIBU,
  HYUNDAI.ELANTRA,
  HYUNDAI.ELANTRA_GT_I30,
  HYUNDAI.GENESIS_G90,
  HYUNDAI.KIA_FORTE,
  HYUNDAI.KIA_OPTIMA_H,
  HYUNDAI.KONA_EV,
]

routes: dict = {
  "0c94aa1e1296d7c6|2021-05-05--19-48-37": {
    'carFingerprint': CHRYSLER.JEEP_CHEROKEE,
  },
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
  },
  "a74b011b32b51b56|2020-07-26--17-09-36": {
    'carFingerprint': HONDA.CIVIC,
    'enableCamera': True,
  },
  "a859a044a447c2b0|2020-03-03--18-42-45": {
    'carFingerprint': HONDA.CRV_EU,
    'enableCamera': True,
  },
  "68aac44ad69f838e|2021-05-18--20-40-52": {
    'carFingerprint': HONDA.CRV,
    'enableCamera': True,
  },
  "14fed2e5fa0aa1a5|2021-05-25--14-59-42": {
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
  },
  "917b074700869333|2021-05-24--20-40-20": {
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
  "1da5847ac2488106|2021-05-24--19-31-50": {
    'carFingerprint': HONDA.ACCORD,
    'enableCamera': True,
  },
  "07585b0da3c88459|2021-05-26--18-52-04": {
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
  },
  "f0890d16a07a236b|2021-05-25--17-27-22": {
    'carFingerprint': HONDA.INSIGHT,
    'enableCamera': True,
  },
  "07d37d27996096b6|2020-03-04--21-57-27": {
    'carFingerprint': HONDA.PILOT,
    'enableCamera': True,
  },
  "fa1cd231131ca137|2021-05-22--07-59-57": {
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
  "54fd8451b3974762|2021-04-01--14-50-10": {
    'carFingerprint': HONDA.RIDGELINE,
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
  "e0e98335f3ebc58f|2021-03-07--16-38-29": {
    'carFingerprint': HYUNDAI.KIA_CEED,
    'enableCamera': True,
  },
  "7653b2bce7bcfdaa|2020-03-04--15-34-32": {
    'carFingerprint': HYUNDAI.KIA_OPTIMA,
    'enableCamera': True,
  },
  "c75a59efa0ecd502|2021-03-11--20-52-55": {
    'carFingerprint': HYUNDAI.KIA_SELTOS,
    'enableCamera': True,
  },
  "5b7c365c50084530|2020-04-15--16-13-24": {
    'carFingerprint': HYUNDAI.SONATA,
    'enableCamera': True,
  },
  "b2a38c712dcf90bd|2020-05-18--18-12-48": {
    'carFingerprint': HYUNDAI.SONATA_LF,
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
  "2c5cf2dd6102e5da|2020-12-17--16-06-44": {
    'carFingerprint': HYUNDAI.IONIQ_EV_2020,
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
  "82e9cdd3f43bf83e|2021-05-15--02-42-51": {
    'carFingerprint': HYUNDAI.ELANTRA_2021,
    'enableCamera': True,
  },
  "000cf3730200c71c|2021-05-24--10-42-05": {
    'carFingerprint': TOYOTA.AVALON,
    'enableCamera': True,
    'enableDsu': False,
  },
  "6cdecc4728d4af37|2020-02-23--15-44-18": {
    'carFingerprint': TOYOTA.CAMRY,
    'enableCamera': True,
    'enableDsu': False,
  },
  "3456ad0cd7281b24|2020-12-13--17-45-56": {
    'carFingerprint': TOYOTA.CAMRY_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "ffccc77938ddbc44|2021-01-04--16-55-41": {
    'carFingerprint': TOYOTA.CAMRYH_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "54034823d30962f5|2021-05-24--06-37-34": {
    'carFingerprint': TOYOTA.CAMRYH,
  },
  "4e45c89c38e8ec4d|2021-05-02--02-49-28": {
    'carFingerprint': TOYOTA.COROLLA,
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
  "d2525c22173da58b|2021-04-25--16-47-04": {
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
  "37041c500fd30100|2020-12-30--12-17-24": {
    'carFingerprint': TOYOTA.LEXUS_ESH,
    'enableCamera': True,
    'enableDsu': True,
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
  "d27ad752e9b08d4f|2021-05-26--19-39-51": {
    'carFingerprint': TOYOTA.LEXUS_RXH,
    'enableCamera': True,
    'enableDsu': True,
  },
  "01b22eb2ed121565|2020-02-02--11-25-51": {
    'carFingerprint': TOYOTA.LEXUS_RX_TSS2,
    'enableCamera': True,
  },
  "b74758c690a49668|2020-05-20--15-58-57": {
    'carFingerprint': TOYOTA.LEXUS_RXH_TSS2,
    'enableCamera': True,
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
  "3fd5305f8b6ca765|2021-04-28--19-26-49": {
    'carFingerprint': TOYOTA.LEXUS_NX_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "f33ec5db3dc94da9|2021-05-22--18-19-45": {
    'carFingerprint': TOYOTA.LEXUS_UXH_TSS2,
  },
  # TODO: missing some combos for highlander
  "0a302ffddbb3e3d3|2020-02-08--16-19-08": {
    'carFingerprint': TOYOTA.HIGHLANDER_TSS2,
    'enableCamera': True,
    'enableDsu': False,
  },
  "437e4d2402abf524|2021-05-25--07-58-50": {
    'carFingerprint': TOYOTA.HIGHLANDERH_TSS2,
  },
  "3183cd9b021e89ce|2021-05-25--10-34-44": {
    'carFingerprint': TOYOTA.HIGHLANDER,
    'enableCamera': True,
  },
  "80d16a262e33d57f|2021-05-23--20-01-43": {
    'carFingerprint': TOYOTA.HIGHLANDERH,
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
  "9b36accae406390e|2021-03-30--10-41-38": {
    'carFingerprint': TOYOTA.MIRAI,
    'enableCamera': True,
    'enableDsu': False,
  },
  "cd9cff4b0b26c435|2021-05-13--15-12-39": {
    'carFingerprint': TOYOTA.CHR,
  },
  "57858ede0369a261|2021-05-18--20-34-20": {
    'carFingerprint': TOYOTA.CHRH,
  },
  "2c68dda277d887ac|2021-05-11--15-22-20": {
    'carFingerprint': VOLKSWAGEN.ATLAS_MK1,
    'enableCamera': True,
  },
  "cae14e88932eb364|2021-03-26--14-43-28": {
    'carFingerprint': VOLKSWAGEN.GOLF_MK7,
    'enableCamera': True,
  },
  "58a7d3b707987d65|2021-03-25--17-26-37": {
    'carFingerprint': VOLKSWAGEN.JETTA_MK7,
    'enableCamera': True,
  },
  "4d134e099430fba2|2021-03-26--00-26-06": {
    'carFingerprint': VOLKSWAGEN.PASSAT_MK8,
    'enableCamera': True,
  },
  "2cef8a0b898f331a|2021-03-25--20-13-57": {
    'carFingerprint': VOLKSWAGEN.TIGUAN_MK2,
    'enableCamera': True,
  },
  "07667b885add75fd|2021-01-23--19-48-42": {
    'carFingerprint': VOLKSWAGEN.AUDI_A3_MK3,
    'enableCamera': True,
  },
  "8f205bdd11bcbb65|2021-03-26--01-00-17": {
    'carFingerprint': VOLKSWAGEN.SEAT_ATECA_MK1,
    'enableCamera': True,
  },
  "90434ff5d7c8d603|2021-03-15--12-07-31": {
    'carFingerprint': VOLKSWAGEN.SKODA_KODIAQ_MK1,
    'enableCamera': True,
  },
  "66e5edc3a16459c5|2021-05-25--19-00-29": {
    'carFingerprint': VOLKSWAGEN.SKODA_OCTAVIA_MK3,
    'enableCamera': True,
  },
  "026b6d18fba6417f|2021-03-26--09-17-04": {
    'carFingerprint': VOLKSWAGEN.SKODA_SCALA_MK1,
    'enableCamera': True,
  },
  "b2e9858e29db492b|2021-03-26--16-58-42": {
    'carFingerprint': VOLKSWAGEN.SKODA_SUPERB_MK3,
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
  "22c3dcce2dd627eb|2020-12-30--16-38-48": {
    'carFingerprint': NISSAN.LEAF_IC,
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
  "b72d3ec617c0a90f|2020-12-11--15-38-17": {
    'carFingerprint': NISSAN.ALTIMA,
    'enableCamera': True,
  },
}

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
