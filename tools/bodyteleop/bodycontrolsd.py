import json
from cereal import messaging

def main():
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['bodyReserved0'])

  while True:
      sm.update(0)

      if sm.updated['bodyReserved0']:
          controls = json.loads(sm.bodyReserved0)
          msg = messaging.new_message('testJoystick')
          msg.testJoystick.axes = [controls['x'], controls['y']]
          msg.testJoystick.buttons = [False]
          pm.send('testJoystick', msg)


if __name__ == "__main__":
  main()
