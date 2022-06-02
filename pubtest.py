import cereal.messaging as messaging

pm = messaging.PubMaster(['sensorEvents'])
dat = messaging.new_message('sensorEvents', size=2)
dat.sensorEvents[0] = {"gyro": {"v": [0.12, -0.22, 0.11]}}
pm.send('sensorEvents', dat)
print("sent")
