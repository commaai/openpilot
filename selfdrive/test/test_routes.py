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
  CHRYSLER.JEEP_CHEROKEE,
  CHRYSLER.JEEP_CHEROKEE_2019,
  CHRYSLER.PACIFICA_2018,
  CHRYSLER.PACIFICA_2018_HYBRID,
  CHRYSLER.PACIFICA_2020,
  GM.CADILLAC_ATS,
  GM.HOLDEN_ASTRA,
  GM.MALIBU,
  HONDA.CRV,
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


class TestRoute:
  def __init__(self, route, car_fingerprint, *, enable_camera=True, enable_dsu=False, enable_gas_interceptor=False):
    self.route = route
    self.car_fingerprint = car_fingerprint
    self.enable_camera = enable_camera
    self.enable_dsu = enable_dsu
    self.enable_gas_interceptor = enable_gas_interceptor


routes = [
  TestRoute("420a8e183f1aed48|2020-03-05--07-15-29", CHRYSLER.PACIFICA_2017_HYBRID),
  TestRoute("8190c7275a24557b|2020-01-29--08-33-58", CHRYSLER.PACIFICA_2019_HYBRID),  # 2020 model year
  TestRoute("f1b4c567731f4a1b|2018-04-18--11-29-37", FORD.FUSION, enable_camera=False),
  TestRoute("f1b4c567731f4a1b|2018-04-30--10-15-35", FORD.FUSION),
  TestRoute("c950e28c26b5b168|2018-05-30--22-03-41", GM.VOLT),
  # TODO: use another route that has radar data at start
  TestRoute("7cc2a8365b4dd8a9|2018-12-02--12-10-44", GM.ACADIA),
  TestRoute("aa20e335f61ba898|2019-02-05--16-59-04", GM.BUICK_REGAL),
  TestRoute("0e7a2ba168465df5|2020-10-18--14-14-22", HONDA.ACURA_RDX_3G),
  TestRoute("a74b011b32b51b56|2020-07-26--17-09-36", HONDA.CIVIC),
  TestRoute("a859a044a447c2b0|2020-03-03--18-42-45", HONDA.CRV_EU),
  TestRoute("232585b7784c1af4|2019-04-08--14-12-14", HONDA.CRV_HYBRID),
  TestRoute("99e3eaed7396619e|2019-08-13--15-07-03", HONDA.FIT),
  TestRoute("03be5f2fd5c508d1|2020-04-19--18-44-15", HONDA.HRV),
  TestRoute("2ac95059f70d76eb|2018-02-05--15-03-29", HONDA.ACURA_ILX),
  TestRoute("81722949a62ea724|2019-03-29--15-51-26", HONDA.ODYSSEY_CHN, enable_camera=False),
  TestRoute("81722949a62ea724|2019-04-06--15-19-25", HONDA.ODYSSEY_CHN),
  TestRoute("08a3deb07573f157|2020-03-06--16-11-19", HONDA.ACCORD_15),
  TestRoute("a8e8bf6a3864361b|2021-04-20--12-09-18", HONDA.ACCORD),
  TestRoute("690c4c9f9f2354c7|2018-09-15--17-36-05", HONDA.ACCORDH),
  TestRoute("1ad763dd22ef1a0e|2020-02-29--18-37-03", HONDA.CRV_5G),
  TestRoute("0a96f86fcfe35964|2020-02-05--07-25-51", HONDA.ODYSSEY),
  TestRoute("d83f36766f8012a5|2020-02-05--18-42-21", HONDA.CIVIC_BOSCH_DIESEL),
  TestRoute("fb51d190ddfd8a90|2020-02-25--14-43-43", HONDA.INSIGHT),
  TestRoute("07d37d27996096b6|2020-03-04--21-57-27", HONDA.PILOT),
  TestRoute("22affd6c545d985e|2020-03-08--01-08-09", HONDA.PILOT_2019),
  TestRoute("0a78dfbacc8504ef|2020-03-04--13-29-55", HONDA.CIVIC_BOSCH),
  TestRoute("f34a60d68d83b1e5|2020-10-06--14-35-55", HONDA.ACURA_RDX),
  TestRoute("54fd8451b3974762|2021-04-01--14-50-10", HONDA.RIDGELINE),
  TestRoute("6fe86b4e410e4c37|2020-07-22--16-27-13", HYUNDAI.HYUNDAI_GENESIS),
  TestRoute("70c5bec28ec8e345|2020-08-08--12-22-23", HYUNDAI.GENESIS_G70),
  TestRoute("6b301bf83f10aa90|2020-11-22--16-45-07", HYUNDAI.GENESIS_G80),
  TestRoute("38bfd238edecbcd7|2018-08-22--09-45-44", HYUNDAI.SANTA_FE, enable_camera=False),
  TestRoute("38bfd238edecbcd7|2018-08-29--22-02-15", HYUNDAI.SANTA_FE),
  TestRoute("e0e98335f3ebc58f|2021-03-07--16-38-29", HYUNDAI.KIA_CEED),
  TestRoute("7653b2bce7bcfdaa|2020-03-04--15-34-32", HYUNDAI.KIA_OPTIMA),
  TestRoute("c75a59efa0ecd502|2021-03-11--20-52-55", HYUNDAI.KIA_SELTOS),
  TestRoute("5b7c365c50084530|2020-04-15--16-13-24", HYUNDAI.SONATA),
  TestRoute("b2a38c712dcf90bd|2020-05-18--18-12-48", HYUNDAI.SONATA_LF),
  TestRoute("5875672fc1d4bf57|2020-07-23--21-33-28", HYUNDAI.KIA_SORENTO),
  TestRoute("9c917ba0d42ffe78|2020-04-17--12-43-19", HYUNDAI.PALISADE),
  TestRoute("2c5cf2dd6102e5da|2020-12-17--16-06-44", HYUNDAI.IONIQ_EV_2020),
  TestRoute("610ebb9faaad6b43|2020-06-13--15-28-36", HYUNDAI.IONIQ_EV_LTD),
  TestRoute("2c5cf2dd6102e5da|2020-06-26--16-00-08", HYUNDAI.IONIQ),
  TestRoute("22d955b2cd499c22|2020-08-10--19-58-21", HYUNDAI.KONA),
  TestRoute("5dddcbca6eb66c62|2020-07-26--13-24-19", HYUNDAI.KIA_STINGER),
  TestRoute("d624b3d19adce635|2020-08-01--14-59-12", HYUNDAI.VELOSTER),
  TestRoute("50c6c9b85fd1ff03|2020-10-26--17-56-06", HYUNDAI.KIA_NIRO_EV),
  TestRoute("f7b6be73e3dfd36c|2019-05-12--18-07-16", TOYOTA.AVALON, enable_camera=False),
  TestRoute("6cdecc4728d4af37|2020-02-23--15-44-18", TOYOTA.CAMRY),
  TestRoute("3456ad0cd7281b24|2020-12-13--17-45-56", TOYOTA.CAMRY_TSS2),
  TestRoute("ffccc77938ddbc44|2021-01-04--16-55-41", TOYOTA.CAMRYH_TSS2),
  TestRoute("f7b6be73e3dfd36c|2019-05-11--22-34-20", TOYOTA.AVALON),
  TestRoute("4e45c89c38e8ec4d|2021-05-02--02-49-28", TOYOTA.COROLLA),
  TestRoute("5f5afb36036506e4|2019-05-14--02-09-54", TOYOTA.COROLLA_TSS2),
  TestRoute("5ceff72287a5c86c|2019-10-19--10-59-02", TOYOTA.COROLLAH_TSS2),
  TestRoute("d2525c22173da58b|2021-04-25--16-47-04", TOYOTA.PRIUS, enable_dsu=True),
  TestRoute("b0f5a01cf604185c|2017-12-18--20-32-32", TOYOTA.RAV4, enable_dsu=True),
  TestRoute("b0c9d2329ad1606b|2019-04-02--13-24-43", TOYOTA.RAV4, enable_dsu=True, enable_gas_interceptor=True),
  TestRoute("b14c5b4742e6fc85|2020-07-28--19-50-11", TOYOTA.RAV4, enable_gas_interceptor=True),
  TestRoute("32a7df20486b0f70|2020-02-06--16-06-50", TOYOTA.RAV4H, enable_dsu=True),
  TestRoute("cdf2f7de565d40ae|2019-04-25--03-53-41", TOYOTA.RAV4_TSS2),
  TestRoute("7e34a988419b5307|2019-12-18--19-13-30", TOYOTA.RAV4H_TSS2),
  TestRoute("e6a24be49a6cd46e|2019-10-29--10-52-42", TOYOTA.LEXUS_ES_TSS2),
  TestRoute("25057fa6a5a63dfb|2020-03-04--08-44-23", TOYOTA.LEXUS_CTH, enable_dsu=True),
  TestRoute("f49e8041283f2939|2019-05-29--13-48-33", TOYOTA.LEXUS_ESH_TSS2, enable_camera=False),
  TestRoute("f49e8041283f2939|2019-05-30--11-51-51", TOYOTA.LEXUS_ESH_TSS2),
  TestRoute("37041c500fd30100|2020-12-30--12-17-24", TOYOTA.LEXUS_ESH, enable_dsu=True),
  TestRoute("886fcd8408d570e9|2020-01-29--05-11-22", TOYOTA.LEXUS_RX, enable_dsu=True),
  TestRoute("886fcd8408d570e9|2020-01-29--02-18-55", TOYOTA.LEXUS_RX),
  TestRoute("b0f5a01cf604185c|2018-02-01--21-12-28", TOYOTA.LEXUS_RXH, enable_dsu=True),
  TestRoute("01b22eb2ed121565|2020-02-02--11-25-51", TOYOTA.LEXUS_RX_TSS2),
  TestRoute("b74758c690a49668|2020-05-20--15-58-57", TOYOTA.LEXUS_RXH_TSS2),
  TestRoute("ec429c0f37564e3c|2020-02-01--17-28-12", TOYOTA.LEXUS_NXH),
  TestRoute("964c09eb11ca8089|2020-11-03--22-04-00", TOYOTA.LEXUS_NX),
  TestRoute("3fd5305f8b6ca765|2021-04-28--19-26-49", TOYOTA.LEXUS_NX_TSS2),
  # TODO: missing some combos for highlander
  TestRoute("0a302ffddbb3e3d3|2020-02-08--16-19-08", TOYOTA.HIGHLANDER_TSS2),
  TestRoute("aa659debdd1a7b54|2018-08-31--11-12-01", TOYOTA.HIGHLANDER, enable_camera=False),
  TestRoute("eb6acd681135480d|2019-06-20--20-00-00", TOYOTA.SIENNA),
  TestRoute("2e07163a1ba9a780|2019-08-25--13-15-13", TOYOTA.LEXUS_IS),
  TestRoute("2e07163a1ba9a780|2019-08-29--09-35-42", TOYOTA.LEXUS_IS, enable_camera=False),
  TestRoute("0a0de17a1e6a2d15|2020-09-21--21-24-41", TOYOTA.PRIUS_TSS2),
  TestRoute("9b36accae406390e|2021-03-30--10-41-38", TOYOTA.MIRAI),
  TestRoute("cae14e88932eb364|2021-03-26--14-43-28", VOLKSWAGEN.GOLF_MK7),
  TestRoute("58a7d3b707987d65|2021-03-25--17-26-37", VOLKSWAGEN.JETTA_MK7),
  TestRoute("4d134e099430fba2|2021-03-26--00-26-06", VOLKSWAGEN.PASSAT_MK8),
  TestRoute("2cef8a0b898f331a|2021-03-25--20-13-57", VOLKSWAGEN.TIGUAN_MK2),
  TestRoute("07667b885add75fd|2021-01-23--19-48-42", VOLKSWAGEN.AUDI_A3_MK3),
  TestRoute("8f205bdd11bcbb65|2021-03-26--01-00-17", VOLKSWAGEN.SEAT_ATECA_MK1),
  TestRoute("90434ff5d7c8d603|2021-03-15--12-07-31", VOLKSWAGEN.SKODA_KODIAQ_MK1),
  TestRoute("026b6d18fba6417f|2021-03-26--09-17-04", VOLKSWAGEN.SKODA_SCALA_MK1),
  TestRoute("b2e9858e29db492b|2021-03-26--16-58-42", VOLKSWAGEN.SKODA_SUPERB_MK3),
  TestRoute("3c8f0c502e119c1c|2020-06-30--12-58-02", SUBARU.ASCENT),
  TestRoute("c321c6b697c5a5ff|2020-06-23--11-04-33", SUBARU.FORESTER),
  TestRoute("791340bc01ed993d|2019-03-10--16-28-08", SUBARU.IMPREZA),
  # Dashcam
  TestRoute("95441c38ae8c130e|2020-06-08--12-10-17", SUBARU.FORESTER_PREGLOBAL),
  # Dashcam
  TestRoute("df5ca7660000fba8|2020-06-16--17-37-19", SUBARU.LEGACY_PREGLOBAL),
  # Dashcam
  TestRoute("5ab784f361e19b78|2020-06-08--16-30-41", SUBARU.OUTBACK_PREGLOBAL),
  # Dashcam
  TestRoute("e19eb5d5353b1ac1|2020-08-09--14-37-56", SUBARU.OUTBACK_PREGLOBAL_2018),
  TestRoute("fbbfa6af821552b9|2020-03-03--08-09-43", NISSAN.XTRAIL),
  TestRoute("5b7c365c50084530|2020-03-25--22-10-13", NISSAN.LEAF),
  TestRoute("22c3dcce2dd627eb|2020-12-30--16-38-48", NISSAN.LEAF_IC),
  TestRoute("059ab9162e23198e|2020-05-30--09-41-01", NISSAN.ROGUE),
  TestRoute("32a319f057902bb3|2020-04-27--15-18-58", MAZDA.CX5),
  TestRoute("10b5a4b380434151|2020-08-26--17-11-45", MAZDA.CX9),
  TestRoute("74f1038827005090|2020-08-26--20-05-50", MAZDA.Mazda3),
  TestRoute("b72d3ec617c0a90f|2020-12-11--15-38-17", NISSAN.ALTIMA),
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
