import pickle
from selfdrive.car.toyota.values import CAR, DBC, CAR_INFO, ToyotaFlags
from selfdrive.car.toyota_old.values import CAR as OLD_CAR, DBC as OLD_DBC, CAR_INFO as OLD_CAR_INFO

# for k, v in DBC.items():
#   if v != OLD_DBC[str(k)]:
#     print(k, v)
#     print(k, OLD_DBC[str(k)])
#     print()
# # assert {str(k): v for k, v in DBC.items()} == OLD_DBC
# for k, v in CAR_INFO.items():
#   if str(v) != str(OLD_CAR_INFO[str(k)]):
#     print(k, v)
#     print(k, OLD_CAR_INFO[str(k)])
#     print()


CPs_OLD = {}
CPs_NEW = {}

for platform in CAR:
  old_platform = str(platform).replace('TOYOTA', 'TOYOTA_OLD').replace('LEXUS', 'LEXUS_OLD')
  with open(f'/home/batman/toyota_data_temp/{str(old_platform)}', 'rb') as f:
    CPs_OLD[platform] = pickle.load(f).as_builder()
    CPs_OLD[platform].carFingerprint = platform
    CPs_OLD[platform] = CPs_OLD[platform].as_reader()
  with open(f'/home/batman/toyota_data_temp/{str(platform)}', 'rb') as f:
    CPs_NEW[platform] = pickle.load(f)

for platform in CAR:
  old_dict = CPs_OLD[platform].to_dict()
  new_dict = CPs_NEW[platform].to_dict()
  # if old_dict != new_dict:
  #   print('differ', platform)
  #   print('old', old_dict)
  #   print('new', new_dict)
  #   print()

  for attr in new_dict.keys():
    if attr == 'flags':
      new_dict[attr] &= ~ToyotaFlags.TSS2
      new_dict[attr] &= ~ToyotaFlags.NO_STOP_TIMER
      new_dict[attr] &= ~ToyotaFlags.ANGLE_CONTROL
      new_dict[attr] &= ~ToyotaFlags.RADAR_ACC
      new_dict[attr] &= ~ToyotaFlags.NO_DSU
      new_dict[attr] &= ~ToyotaFlags.UNSUPPORTED_DSU
    if old_dict[attr] != new_dict[attr]:
      print('differ', platform, attr)
      if attr == 'flags':
        print('old', repr(ToyotaFlags(old_dict[attr])))
        print('new', repr(ToyotaFlags(new_dict[attr])))
      else:
        print('old', old_dict[attr])
        print('new', new_dict[attr])
      print()

    # if CPs_OLD[platform].to_dict()[attr] != getattr(CPs_NEW[platform], attr):
    #   print('differ', platform, attr)
    #   print('old', getattr(CPs_OLD[platform], attr))
    #   print('new', getattr(CPs_NEW[platform], attr))
    #   print()
  # break
#
