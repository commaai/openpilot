import numpy as np


MIN_ACC_SPEED = 8.49376

orig_kpBP = [0., 5., 35.]
orig_kiBP = [0., 35.]

stock_kpV = [3.6, 2.4, 1.5]
stock_kiV = [0.54, 0.36]

pedal_kpV = [1.2, 0.8, 0.5]
pedal_kiV = [0.18, 0.12]

hybrid_kpBP = [0., 5., MIN_ACC_SPEED, MIN_ACC_SPEED, 35.]
hybrid_kiBP = [0., MIN_ACC_SPEED, MIN_ACC_SPEED, 35.]

hybrid_kpV = pedal_kpV[:2]  # all v's below min acc speed
hybrid_kpV.append(np.interp(MIN_ACC_SPEED, orig_kpBP, pedal_kpV))
hybrid_kpV.append(np.interp(MIN_ACC_SPEED, orig_kpBP, stock_kpV))
hybrid_kpV.append(stock_kpV[-1])  # all v's above MIN_ACC_SPEED
print(np.round(hybrid_kpV, 3).tolist())

hybrid_kiV = pedal_kiV[:1]  # all v's below min acc speed
hybrid_kiV.append(np.interp(MIN_ACC_SPEED, orig_kiBP, pedal_kiV))
hybrid_kiV.append(np.interp(MIN_ACC_SPEED, orig_kiBP, stock_kiV))
hybrid_kiV.append(stock_kiV[-1])  # all v's above MIN_ACC_SPEED
print(np.round(hybrid_kiV, 3).tolist())


