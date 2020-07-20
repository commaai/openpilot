#!/usr/bin/env python3
from selfdrive.car.fingerprints import eliminate_incompatible_cars, all_known_cars
import cereal.messaging as messaging


# rav4 2019 and corolla tss2
fingerprint = {896: 8, 898: 8, 900: 6, 976: 1, 1541: 8, 902: 6, 905: 8, 810: 2, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1552: 8, 1553: 8, 1556: 8, 1571: 8, 921: 8, 1056: 8, 544: 4, 1570: 8, 1059: 1, 36: 8, 37: 8, 550: 8, 935: 8, 552: 4, 170: 8, 812: 8, 944: 8, 945: 8, 562: 6, 180: 8, 1077: 8, 951: 8, 1592: 8, 1076: 8, 186: 4, 955: 8, 956: 8, 1001: 8, 705: 8, 452: 8, 1788: 8, 464: 8, 824: 8, 466: 8, 467: 8, 761: 8, 728: 8, 1572: 8, 1114: 8, 933: 8, 800: 8, 608: 8, 865: 8, 610: 8, 1595: 8, 934: 8, 998: 5, 1745: 8, 1000: 8, 764: 8, 1002: 8, 999: 7, 1789: 8, 1649: 8, 1779: 8, 1568: 8, 1017: 8, 1786: 8, 1787: 8, 1020: 8, 426: 6, 1279: 8}

candidate_cars = all_known_cars()


for addr, l in fingerprint.items():
    dat = messaging.new_message('can', 1)

    msg = dat.can[0]
    msg.address = addr
    msg.dat = " " * l

    candidate_cars = eliminate_incompatible_cars(msg, candidate_cars)
    print(candidate_cars)
