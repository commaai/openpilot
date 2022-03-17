import cereal.messaging as messaging
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate
import matplotlib.patches as mpatches

topics = ['roadCameraState', 'modelV2']
frames = defaultdict(lambda: defaultdict(int))

sm = messaging.SubMaster(topics)
while 1:
    sm.update()
    '''
    for topic, is_updated in sm.updated.items():
        if is_updated:
            sm[topic].frameId
            '''
    print("-------------------")
    print(sm['roadCameraState'].frameId, [topic for topic, val in sm.updated.items() if val])
    print(sm['modelV2'].frameId, [topic for topic, val in sm.updated.items() if val])


