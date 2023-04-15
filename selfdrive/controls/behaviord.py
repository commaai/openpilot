#!/usr/bin/env python3

""" gets behavior input from UI and saves it to params"""
from common.params import Params
import cereal.messaging as messaging
import time





class Behaviord:
  def __init__(self):
    
    sm = messaging.SubMaster(['behavior'])
    self.sm = sm
    self.p = Params()
    self.init_params()

  
  def init_params(self):
    if self.p.get("ComfortBrake") is None:
      self.p.put("ComfortBrake", "0")

  def save(self):
    if self.sm.updated['behavior']:
      print("behavior updated")
      self.p.put("ComfortBrake", str(self.sm['behavior'].comfortBrake))
      # Add more params here

  def behaviord_thread(self):
    while 1:
      self.sm.update(0)
      self.save()
      #self.send()
      time.sleep(1)

def main():
  behaviord = Behaviord()
  behaviord.behaviord_thread() 
            
if __name__ == "__main__":
  main()
            