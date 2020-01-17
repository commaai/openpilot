#!/usr/bin/env python3
import zmq
import cereal.messaging as messaging
from cereal.services import service_list
import pcap

def main(gctx=None):
  ethernetData = messaging.pub_sock('ethernetData')

  for ts, pkt in pcap.pcap('eth0'):
    dat = messaging.new_message()
    dat.init('ethernetData', 1)
    dat.ethernetData[0].ts = ts
    dat.ethernetData[0].pkt = str(pkt)
    ethernetData.send(dat.to_bytes())

if __name__ == "__main__":
  main()

